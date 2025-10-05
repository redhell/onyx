"""
Unit tests for fast_chat_turn functionality.

This module contains unit tests for the fast_chat_turn function, which handles
chat turn processing with agent-based interactions. The tests use dependency
injection with simple fake versions of all dependencies except for the emitter
(which is created by the unified_event_stream decorator) and dependencies_to_maybe_remove
(which should be passed in by the test writer).
"""

from collections.abc import AsyncIterator
from typing import List
from uuid import uuid4

import pytest
from agents import AgentOutputSchemaBase
from agents import FunctionTool
from agents import Handoff
from agents import Model
from agents import ModelResponse
from agents import ModelSettings
from agents import ModelTracing
from agents import Tool
from agents import Usage
from agents.items import ResponseOutputMessage
from agents.items import ResponseOutputText
from openai.types.responses import Response
from openai.types.responses import ResponseCustomToolCallInputDeltaEvent
from openai.types.responses.response_stream_event import ResponseCompletedEvent
from openai.types.responses.response_stream_event import ResponseCreatedEvent
from openai.types.responses.response_stream_event import ResponseTextDeltaEvent
from openai.types.responses.response_usage import InputTokensDetails
from openai.types.responses.response_usage import OutputTokensDetails
from openai.types.responses.response_usage import ResponseUsage

from onyx.agents.agent_search.dr.enums import ResearchType
from onyx.chat.turn.models import ChatTurnDependencies
from onyx.chat.turn.models import DependenciesToMaybeRemove
from onyx.llm.interfaces import LLM
from onyx.llm.interfaces import LLMConfig
from onyx.server.query_and_chat.streaming_models import OverallStop
from onyx.server.query_and_chat.streaming_models import Packet


# =============================================================================
# Helper Functions and Base Classes for DRY Principles
# =============================================================================


def create_fake_usage() -> Usage:
    """Create a standard fake usage object."""
    return Usage(
        requests=1,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def create_fake_response_usage() -> ResponseUsage:
    """Create a standard fake response usage object."""
    return ResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def create_fake_message(
    text: str = "fake response", include_tool_calls: bool = False
) -> ResponseOutputMessage:
    """Create a fake response message with optional tool calls."""
    content = [ResponseOutputText(text=text, type="output_text", annotations=[])]

    tool_calls = None
    if include_tool_calls:
        tool_calls = [
            {
                "id": "fake-tool-call-id",
                "type": "function",
                "function": {
                    "name": "fake_cancellation_tool",
                    "arguments": '{"query": "test query"}',
                },
            }
        ]

    return ResponseOutputMessage(
        id="fake-message-id",
        role="assistant",
        content=content,
        status="completed",
        type="message",
        tool_calls=tool_calls,
    )


def create_fake_response(
    response_id: str = "fake-response-id", message: ResponseOutputMessage = None
) -> Response:
    """Create a fake response object."""
    if message is None:
        message = create_fake_message()

    return Response(
        id=response_id,
        created_at=1234567890,
        object="response",
        output=[message],
        usage=create_fake_response_usage(),
        status="completed",
        model="fake-model",
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class BaseFakeModel(Model):
    """Base class for fake models with common functionality."""

    def __init__(
        self, name: str = "fake-model", provider: str = "fake-provider", **kwargs
    ):
        self.name = name
        self.provider = provider
        # Store any additional kwargs for subclasses
        self._kwargs = kwargs

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: List[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt=None,
    ) -> ModelResponse:
        """Default get_response implementation."""
        message = create_fake_message()
        usage = create_fake_usage()
        return ModelResponse(
            output=[message], usage=usage, response_id="fake-response-id"
        )


class StreamableFakeModel(BaseFakeModel):
    """Base class for fake models that support streaming."""

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: List[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt=None,
    ) -> AsyncIterator[object]:
        """Default streaming implementation."""
        return self._create_stream_events()

    def _create_stream_events(
        self,
        message: ResponseOutputMessage = None,
        response_id: str = "fake-response-id",
    ) -> AsyncIterator[object]:
        """Create standard stream events."""

        async def _gen() -> AsyncIterator[object]:
            # Create message if not provided
            msg = message if message is not None else create_fake_message()

            final_response = create_fake_response(response_id, msg)

            # 1) created
            yield ResponseCreatedEvent(
                response=final_response, sequence_number=1, type="response.created"
            )

            # 2) stream some text (delta)
            for _ in range(5):
                yield ResponseTextDeltaEvent(
                    content_index=0,
                    delta="fake response",
                    item_id="fake-item-id",
                    logprobs=[],
                    output_index=0,
                    sequence_number=2,
                    type="response.output_text.delta",
                )

            # 3) completed
            yield ResponseCompletedEvent(
                response=final_response, sequence_number=3, type="response.completed"
            )

        return _gen()


class CancellationMixin:
    """Mixin for models that support cancellation testing."""

    def __init__(
        self, set_fence_func=None, chat_session_id=None, redis_client=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.set_fence_func = set_fence_func
        self.chat_session_id = chat_session_id
        self.redis_client = redis_client

    def _should_trigger_cancellation(self, iteration: int) -> bool:
        """Check if cancellation should be triggered at this iteration."""
        return (
            iteration == 2
            and self.set_fence_func
            and self.chat_session_id
            and self.redis_client
        )

    def _trigger_cancellation(self):
        """Trigger the cancellation signal."""
        if self.set_fence_func and self.chat_session_id and self.redis_client:
            self.set_fence_func(self.chat_session_id, self.redis_client, True)


# =============================================================================
# Test Helper Functions
# =============================================================================


def run_fast_chat_turn(
    sample_messages: list[dict], chat_turn_dependencies: ChatTurnDependencies
) -> list[Packet]:
    """Helper function to run fast_chat_turn and collect all packets."""
    from onyx.chat.turn.fast_chat_turn import fast_chat_turn

    generator = fast_chat_turn(sample_messages, chat_turn_dependencies)
    return list(generator)


def assert_packets_contain_stop(packets: list[Packet]) -> None:
    """Assert that packets contain an OverallStop packet at the end."""
    assert len(packets) >= 1, f"Expected at least 1 packet, got {len(packets)}"
    assert packets[-1] == Packet(
        ind=0, obj=OverallStop(type="stop")
    ), "Last packet should be OverallStop"


def assert_cancellation_packets(
    packets: list[Packet], expect_cancelled_message: bool = False
) -> None:
    """Assert packets after cancellation contain expected structure."""
    min_expected = 3 if expect_cancelled_message else 2
    assert (
        len(packets) >= min_expected
    ), f"Expected at least {min_expected} packets after cancellation, got {len(packets)}"

    # Last packet should be OverallStop
    assert packets[-1].obj.type == "stop", "Last packet should be OverallStop"

    # Second-to-last should be SectionEnd
    assert (
        packets[-2].obj.type == "section_end"
    ), "Second-to-last packet should be SectionEnd"

    # If expecting cancelled message, third-to-last should be MessageStart with "Cancelled"
    if expect_cancelled_message:
        assert (
            packets[-3].obj.type == "message_start"
        ), "Third-to-last packet should be MessageStart"
        from onyx.server.query_and_chat.streaming_models import MessageStart

        assert isinstance(
            packets[-3].obj, MessageStart
        ), "Third-to-last packet should be MessageStart instance"
        assert (
            packets[-3].obj.content == "Cancelled"
        ), "MessageStart should contain 'Cancelled'"


def create_cancellation_model(
    model_class, chat_turn_dependencies: ChatTurnDependencies
):
    """Helper to create a cancellation model with proper setup."""
    from onyx.chat.stop_signal_checker import set_fence

    return model_class(
        set_fence_func=set_fence,
        chat_session_id=chat_turn_dependencies.dependencies_to_maybe_remove.chat_session_id,
        redis_client=chat_turn_dependencies.redis_client,
    )


class FakeLLM(LLM):
    """Simple fake LLM implementation for testing."""

    def __init__(self):
        self._config = LLMConfig(
            model_provider="fake",
            model_name="fake-model",
            temperature=0.7,
            max_input_tokens=4096,
        )

    @property
    def config(self) -> LLMConfig:
        """Return the LLM configuration."""
        return self._config

    def log_model_configs(self) -> None:
        """Fake log_model_configs method."""

    def _invoke_implementation(
        self,
        prompt,
        tools=None,
        tool_choice=None,
        structured_response_format=None,
        timeout_override=None,
        max_tokens=None,
    ):
        """Fake _invoke_implementation method."""
        from langchain_core.messages import AIMessage

        return AIMessage(content="fake response")

    def _stream_implementation(
        self,
        prompt,
        tools=None,
        tool_choice=None,
        structured_response_format=None,
        timeout_override=None,
        max_tokens=None,
    ):
        """Fake _stream_implementation method that yields no messages."""
        return iter([])


class FakeModel(StreamableFakeModel):
    """Simple fake Model implementation for testing Agents SDK."""


class FakeCancellationModel(CancellationMixin, StreamableFakeModel):
    """Fake Model that allows triggering stop signal during streaming."""

    def _create_stream_events(
        self,
        message: ResponseOutputMessage = None,
        response_id: str = "fake-response-id",
    ) -> AsyncIterator[object]:
        """Create stream events with cancellation support."""

        async def _gen() -> AsyncIterator[object]:
            # Create message if not provided
            msg = message if message is not None else create_fake_message()

            final_response = create_fake_response(response_id, msg)

            # 1) created
            yield ResponseCreatedEvent(
                response=final_response, sequence_number=1, type="response.created"
            )

            # 2) stream some text (delta) - trigger stop signal during streaming
            for i in range(5):
                yield ResponseTextDeltaEvent(
                    content_index=0,
                    delta="fake response",
                    item_id="fake-item-id",
                    logprobs=[],
                    output_index=0,
                    sequence_number=2,
                    type="response.output_text.delta",
                )

                # Trigger stop signal after a few deltas
                if self._should_trigger_cancellation(i):
                    self._trigger_cancellation()

            # 3) completed
            yield ResponseCompletedEvent(
                response=final_response, sequence_number=3, type="response.completed"
            )

        return _gen()


class FakeToolCallModel(CancellationMixin, StreamableFakeModel):
    """Fake Model that forces tool calls for testing tool cancellation."""

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: List[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt=None,
    ) -> ModelResponse:
        """Override to create a response with tool calls."""
        message = create_fake_message(
            text="I need to use a tool", include_tool_calls=True
        )
        usage = create_fake_usage()
        return ModelResponse(
            output=[message], usage=usage, response_id="fake-response-id"
        )

    def _create_stream_events(
        self,
        message: ResponseOutputMessage = None,
        response_id: str = "fake-response-id",
    ) -> AsyncIterator[object]:
        """Create stream events with tool calls and cancellation support."""

        async def _gen() -> AsyncIterator[object]:
            # Create message if not provided
            msg = (
                message
                if message is not None
                else create_fake_message(
                    text="I need to use a tool", include_tool_calls=True
                )
            )

            final_response = create_fake_response(response_id, msg)

            # 1) created
            yield ResponseCreatedEvent(
                response=final_response, sequence_number=1, type="response.created"
            )

            # 2) stream tool call deltas - trigger stop signal during streaming
            for i in range(5):
                yield ResponseCustomToolCallInputDeltaEvent(
                    delta="fake response",
                    item_id="fake-item-id",
                    output_index=0,
                    sequence_number=2,
                    type="response.custom_tool_call_input.delta",
                )

                # Trigger stop signal after a few deltas
                if self._should_trigger_cancellation(i):
                    self._trigger_cancellation()

            # 3) completed with the full Response object (including tool calls)
            yield ResponseCompletedEvent(
                response=final_response, sequence_number=2, type="response.completed"
            )

        return _gen()


class FakeFailingModel(BaseFakeModel):
    """Simple fake Model implementation for testing exceptions."""

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: List[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: List[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt=None,
    ) -> AsyncIterator[object]:
        """Stream implementation that raises an exception."""

        async def _gen() -> AsyncIterator[object]:
            yield ResponseCreatedEvent(
                response="hi", sequence_number=1, type="response.created"
            )

            # Stream some deltas before failing
            for i in range(5):
                yield ResponseCustomToolCallInputDeltaEvent(
                    delta="fake response",
                    item_id="fake-item-id",
                    output_index=0,
                    sequence_number=2,
                    type="response.custom_tool_call_input.delta",
                )

            # Raise exception to test error handling
            raise Exception("Fake exception")

        return _gen()


class FakeSession:
    """Simple fake SQLAlchemy Session for testing."""

    def __init__(self):
        self.committed = False
        self.rolled_back = False

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def add(self, instance):
        pass

    def flush(self):
        pass

    def query(self, *args, **kwargs):
        return FakeQuery()

    def execute(self, *args, **kwargs):
        return FakeResult()


class FakeQuery:
    """Simple fake SQLAlchemy Query for testing."""

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        # Return a fake chat message to avoid the "Chat message with id not found" error
        class FakeChatMessage:
            def __init__(self):
                self.id = 123
                self.chat_session_id = "fake-session-id"
                self.message = "fake message"
                self.message_type = "user"
                self.token_count = 0
                self.rephrased_query = None
                self.citations = {}
                self.error = None
                self.alternate_assistant_id = None
                self.overridden_model = None
                self.research_type = "FAST"
                self.research_plan = {}
                self.final_documents = []
                self.research_answer_purpose = "ANSWER"
                self.parent_message = None
                self.is_agentic = False
                self.search_docs = []

        return FakeChatMessage()

    def all(self):
        return []


class FakeResult:
    """Simple fake SQLAlchemy Result for testing."""

    def scalar(self):
        return None

    def fetchall(self):
        return []


class FakeRedis:
    """Simple fake Redis client for testing."""

    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

    def delete(self, key):
        return self.data.pop(key, 0)

    def exists(self, key):
        return key in self.data


@pytest.fixture
def fake_llm() -> LLM:
    """Fixture providing a fake LLM implementation."""
    return FakeLLM()


@pytest.fixture
def fake_model() -> Model:
    """Fixture providing a fake Model implementation."""
    return FakeModel()


@pytest.fixture
def fake_db_session() -> FakeSession:
    """Fixture providing a fake database session."""
    return FakeSession()


@pytest.fixture
def fake_redis_client() -> FakeRedis:
    """Fixture providing a fake Redis client."""
    return FakeRedis()


@pytest.fixture
def fake_tools() -> list[FunctionTool]:
    """Fixture providing a list of fake tools."""
    return []


@pytest.fixture
def dependencies_to_maybe_remove() -> DependenciesToMaybeRemove:
    """Fixture providing dependencies that should be passed in by test writer."""
    return DependenciesToMaybeRemove(
        chat_session_id=uuid4(),
        message_id=123,
        research_type=ResearchType.FAST,
    )


@pytest.fixture
def chat_turn_dependencies(
    fake_llm: LLM,
    fake_model: Model,
    fake_db_session: FakeSession,
    fake_tools: list[FunctionTool],
    fake_redis_client: FakeRedis,
    dependencies_to_maybe_remove: DependenciesToMaybeRemove,
) -> ChatTurnDependencies:
    """Fixture providing a complete ChatTurnDependencies object with fake implementations.

    Note: The emitter field is left as None - it will be set by the unified_event_stream decorator.
    """
    return ChatTurnDependencies(
        llm_model=fake_model,
        llm=fake_llm,
        db_session=fake_db_session,
        tools=fake_tools,
        redis_client=fake_redis_client,
        emitter=None,
        dependencies_to_maybe_remove=dependencies_to_maybe_remove,
    )


@pytest.fixture
def fake_failing_model() -> Model:
    return FakeFailingModel()


@pytest.fixture
def fake_tool_call_model() -> Model:
    return FakeToolCallModel()


@pytest.fixture
def sample_messages() -> list[dict]:
    """Fixture providing sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]


def test_fast_chat_turn_basic(
    chat_turn_dependencies: ChatTurnDependencies,
    sample_messages: list[dict],
):
    """Test that makes sure basic end to end functionality of our
    fast agent chat turn works.
    """
    packets = run_fast_chat_turn(sample_messages, chat_turn_dependencies)
    assert_packets_contain_stop(packets)


def test_fast_chat_turn_catch_exception(
    chat_turn_dependencies: ChatTurnDependencies,
    sample_messages: list[dict],
    fake_failing_model: Model,
):
    """Test that makes sure exceptions in our agent background thread are propagated properly.
    RuntimeWarning: coroutine 'FakeFailingModel.stream_response.<locals>._gen' was never awaited
    is expected.
    """
    from onyx.chat.turn.fast_chat_turn import fast_chat_turn

    chat_turn_dependencies.llm_model = fake_failing_model

    generator = fast_chat_turn(sample_messages, chat_turn_dependencies)
    with pytest.raises(Exception):
        list(generator)


def test_fast_chat_turn_cancellation(
    chat_turn_dependencies: ChatTurnDependencies,
    sample_messages: list[dict],
):
    """Test that cancellation via set_fence works correctly.

    When set_fence is called during message streaming, we should see:
    1. SectionEnd packet (when cancelling during message streaming, no "Cancelled" message is shown)
    2. OverallStop packet

    The "Cancelled" MessageStart is only shown when cancelling during tool calls or reasoning,
    not during regular message streaming.
    """
    # Replace the model with our cancellation model that triggers stop signal during streaming
    cancellation_model = create_cancellation_model(
        FakeCancellationModel, chat_turn_dependencies
    )
    chat_turn_dependencies.llm_model = cancellation_model

    packets = run_fast_chat_turn(sample_messages, chat_turn_dependencies)

    # After cancellation during message streaming, we should see SectionEnd, then OverallStop
    # The "Cancelled" MessageStart is only shown when cancelling during tool calls/reasoning
    assert_cancellation_packets(packets, expect_cancelled_message=False)


def test_fast_chat_turn_tool_call_cancellation(
    chat_turn_dependencies: ChatTurnDependencies,
    sample_messages: list[dict],
):
    """Test that cancellation via set_fence works correctly during tool calls.

    When set_fence is called during tool execution, we should see:
    1. MessageStart packet with "Cancelled" content
    2. SectionEnd packet
    3. OverallStop packet
    """
    # Replace the model with our tool call model
    cancellation_model = create_cancellation_model(
        FakeToolCallModel, chat_turn_dependencies
    )
    chat_turn_dependencies.llm_model = cancellation_model

    packets = run_fast_chat_turn(sample_messages, chat_turn_dependencies)

    # After cancellation during tool call, we should see MessageStart, SectionEnd, then OverallStop
    # The "Cancelled" MessageStart is shown when cancelling during tool calls/reasoning
    assert_cancellation_packets(packets, expect_cancelled_message=True)

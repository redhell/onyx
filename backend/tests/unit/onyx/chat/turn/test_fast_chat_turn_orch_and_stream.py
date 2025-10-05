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


class FakeModel(Model):
    """Simple fake Model implementation for testing Agents SDK."""

    def __init__(self) -> None:
        self.name = "fake-model"
        self.provider = "fake-provider"

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
        # Build a minimal full response (non-streaming path)
        msg = ResponseOutputMessage(
            id="fake-message-id",
            role="assistant",
            content=[
                ResponseOutputText(
                    text="fake response", type="output_text", annotations=[]
                )
            ],
            status="completed",
            type="message",
        )

        usage = Usage(
            requests=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )

        # The ModelResponse wrapper is what the SDK expects here
        return ModelResponse(output=[msg], usage=usage, response_id="fake-response-id")

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
        # Minimal valid sequence of OpenAI Responses stream events.
        async def _gen() -> AsyncIterator[object]:
            response_id = "fake-response-id"

            # Build the response object first
            msg = ResponseOutputMessage(
                id="fake-message-id",
                role="assistant",
                content=[
                    ResponseOutputText(
                        text="fake response", type="output_text", annotations=[]
                    )
                ],
                status="completed",
                type="message",
            )

            usage = ResponseUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            )

            final_response = Response(
                id=response_id,
                created_at=1234567890,
                object="response",
                output=[msg],
                usage=usage,
                status="completed",
                model=self.name,
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            )

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

            # 3) completed with the full Response object
            yield ResponseCompletedEvent(
                response=final_response, sequence_number=3, type="response.completed"
            )

        return _gen()


class FakeCancellationModel(Model):
    """Fake Model that allows triggering stop signal during streaming."""

    def __init__(
        self, set_fence_func=None, chat_session_id=None, redis_client=None
    ) -> None:
        self.name = "fake-model"
        self.provider = "fake-provider"
        self.set_fence_func = set_fence_func
        self.chat_session_id = chat_session_id
        self.redis_client = redis_client

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
        # Build a minimal full response (non-streaming path)
        msg = ResponseOutputMessage(
            id="fake-message-id",
            role="assistant",
            content=[
                ResponseOutputText(
                    text="fake response", type="output_text", annotations=[]
                )
            ],
            status="completed",
            type="message",
        )

        usage = Usage(
            requests=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )

        return ModelResponse(output=[msg], usage=usage, response_id="fake-response-id")

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
        async def _gen() -> AsyncIterator[object]:
            response_id = "fake-response-id"

            # Build the response object first
            msg = ResponseOutputMessage(
                id="fake-message-id",
                role="assistant",
                content=[
                    ResponseOutputText(
                        text="fake response", type="output_text", annotations=[]
                    )
                ],
                status="completed",
                type="message",
            )

            usage = ResponseUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            )

            final_response = Response(
                id=response_id,
                created_at=1234567890,
                object="response",
                output=[msg],
                usage=usage,
                status="completed",
                model=self.name,
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            )

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

                # Trigger stop signal after a few deltas to ensure there are message_delta packets in history
                if (
                    i == 2
                    and self.set_fence_func
                    and self.chat_session_id
                    and self.redis_client
                ):
                    self.set_fence_func(self.chat_session_id, self.redis_client, True)

            # 3) completed with the full Response object
            yield ResponseCompletedEvent(
                response=final_response, sequence_number=3, type="response.completed"
            )

        return _gen()


class FakeToolCallModel(Model):
    """Fake Model that forces tool calls for testing tool cancellation."""

    def __init__(
        self, set_fence_func=None, chat_session_id=None, redis_client=None
    ) -> None:
        self.name = "fake-model"
        self.provider = "fake-provider"
        self.set_fence_func = set_fence_func
        self.chat_session_id = chat_session_id
        self.redis_client = redis_client

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
        # Build a response with tool calls
        msg = ResponseOutputMessage(
            id="fake-message-id",
            role="assistant",
            content=[
                ResponseOutputText(
                    text="I need to use a tool", type="output_text", annotations=[]
                )
            ],
            status="completed",
            type="message",
            tool_calls=[
                {
                    "id": "fake-tool-call-id",
                    "type": "function",
                    "function": {
                        "name": "fake_cancellation_tool",
                        "arguments": '{"query": "test query"}',
                    },
                }
            ],
        )

        usage = Usage(
            requests=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )

        return ModelResponse(output=[msg], usage=usage, response_id="fake-response-id")

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
        async def _gen() -> AsyncIterator[object]:
            response_id = "fake-response-id"

            # Build the response object with tool calls
            msg = ResponseOutputMessage(
                id="fake-message-id",
                role="assistant",
                content=[
                    ResponseOutputText(
                        text="I need to use a tool", type="output_text", annotations=[]
                    )
                ],
                status="completed",
                type="message",
                tool_calls=[
                    {
                        "id": "fake-tool-call-id",
                        "type": "function",
                        "function": {
                            "name": "fake_cancellation_tool",
                            "arguments": '{"query": "test query"}',
                        },
                    }
                ],
            )

            usage = ResponseUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            )

            final_response = Response(
                id=response_id,
                created_at=1234567890,
                object="response",
                output=[msg],
                usage=usage,
                status="completed",
                model=self.name,
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            )

            # 1) created
            yield ResponseCreatedEvent(
                response=final_response, sequence_number=1, type="response.created"
            )

            # 2) stream some text (delta) - trigger stop signal during streaming
            for i in range(5):
                yield ResponseCustomToolCallInputDeltaEvent(
                    delta="fake response",
                    item_id="fake-item-id",
                    output_index=0,
                    sequence_number=2,
                    type="response.custom_tool_call_input.delta",
                )

                # Trigger stop signal after a few deltas to ensure there are message_delta packets in history
                if (
                    i == 2
                    and self.set_fence_func
                    and self.chat_session_id
                    and self.redis_client
                ):
                    self.set_fence_func(self.chat_session_id, self.redis_client, True)

            # 2) completed with the full Response object (including tool calls)
            yield ResponseCompletedEvent(
                response=final_response, sequence_number=2, type="response.completed"
            )

        return _gen()


class FakeFailingModel(Model):
    """Simple fake Model implementation for testing."""

    def __init__(self):
        self.name = "fake-model"
        self.provider = "fake-provider"

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list,
        output_schema,
        handoffs: list,
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt=None,
    ) -> ModelResponse:
        """Fake get_response method that returns a simple response."""
        # Create a simple text response
        text_content = ResponseOutputText(
            text="fake response", type="output_text", annotations=[]
        )
        message = ResponseOutputMessage(
            id="fake-message-id",
            role="assistant",
            content=[text_content],
            status="completed",
            type="message",
        )

        # Create usage information
        usage = Usage(
            requests=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )

        return ModelResponse(
            output=[message], usage=usage, response_id="fake-response-id"
        )

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
        async def _gen() -> AsyncIterator[object]:
            yield ResponseCreatedEvent(
                response="hi", sequence_number=1, type="response.created"
            )

            # 2) stream some text (delta) - trigger stop signal during streaming
            for i in range(5):
                yield ResponseCustomToolCallInputDeltaEvent(
                    delta="fake response",
                    item_id="fake-item-id",
                    output_index=0,
                    sequence_number=2,
                    type="response.custom_tool_call_input.delta",
                )

                # Trigger stop signal after a few deltas to ensure there are message_delta packets in history
                if (
                    i == 2
                    and self.set_fence_func
                    and self.chat_session_id
                    and self.redis_client
                ):
                    self.set_fence_func(self.chat_session_id, self.redis_client, True)
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
    from onyx.chat.turn.fast_chat_turn import fast_chat_turn

    generator = fast_chat_turn(sample_messages, chat_turn_dependencies)
    packets = list(generator)
    # The test should end with an OverallStop packet, but may have other packets before it
    assert len(packets) >= 1
    assert packets[-1] == Packet(ind=0, obj=OverallStop(type="stop"))


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
    monkeypatch,
):
    """Test that cancellation via set_fence works correctly.

    When set_fence is called during message streaming, we should see:
    1. SectionEnd packet (when cancelling during message streaming, no "Cancelled" message is shown)
    2. OverallStop packet

    The "Cancelled" MessageStart is only shown when cancelling during tool calls or reasoning,
    not during regular message streaming.
    """
    from onyx.chat.stop_signal_checker import set_fence
    from onyx.chat.turn.fast_chat_turn import fast_chat_turn

    # Mock get_redis_client to return our fake redis client
    # This is needed because set_fence and is_connected use get_redis_client() directly
    monkeypatch.setattr(
        "onyx.chat.stop_signal_checker.get_redis_client",
        lambda: chat_turn_dependencies.redis_client,
    )

    # Mock get_current_tenant_id to return a test tenant ID
    monkeypatch.setattr(
        "onyx.chat.stop_signal_checker.get_current_tenant_id", lambda: "test-tenant"
    )

    # Replace the model with our cancellation model that triggers stop signal during streaming
    cancellation_model = FakeCancellationModel(
        set_fence_func=set_fence,
        chat_session_id=chat_turn_dependencies.dependencies_to_maybe_remove.chat_session_id,
        redis_client=chat_turn_dependencies.redis_client,
    )
    chat_turn_dependencies.llm_model = cancellation_model

    generator = fast_chat_turn(sample_messages, chat_turn_dependencies)

    packets = []
    for packet in generator:
        packets.append(packet)

    # After cancellation during message streaming, we should see SectionEnd, then OverallStop
    # The "Cancelled" MessageStart is only shown when cancelling during tool calls/reasoning
    assert (
        len(packets) >= 2
    ), f"Expected at least 2 packets after cancellation, got {len(packets)}"

    # The last packet should be OverallStop
    assert packets[-1].obj.type == "stop", "Last packet should be OverallStop"

    # The second-to-last should be SectionEnd
    assert (
        packets[-2].obj.type == "section_end"
    ), "Second-to-last packet should be SectionEnd"


def test_fast_chat_turn_tool_call_cancellation(
    chat_turn_dependencies: ChatTurnDependencies,
    sample_messages: list[dict],
    monkeypatch,
):
    """Test that cancellation via set_fence works correctly during tool calls.

    When set_fence is called during tool execution, we should see:
    1. MessageStart packet with "Cancelled" content
    2. SectionEnd packet
    3. OverallStop packet
    """
    from onyx.chat.stop_signal_checker import set_fence
    from onyx.chat.turn.fast_chat_turn import fast_chat_turn
    from onyx.server.query_and_chat.streaming_models import MessageStart

    # Mock get_redis_client to return our fake redis client
    # This is needed because set_fence and is_connected use get_redis_client() directly
    monkeypatch.setattr(
        "onyx.chat.stop_signal_checker.get_redis_client",
        lambda: chat_turn_dependencies.redis_client,
    )

    # Mock get_current_tenant_id to return a test tenant ID
    monkeypatch.setattr(
        "onyx.chat.stop_signal_checker.get_current_tenant_id", lambda: "test-tenant"
    )

    # Replace the model with our tool call model
    cancellation_model = FakeToolCallModel(
        set_fence_func=set_fence,
        chat_session_id=chat_turn_dependencies.dependencies_to_maybe_remove.chat_session_id,
        redis_client=chat_turn_dependencies.redis_client,
    )
    chat_turn_dependencies.llm_model = cancellation_model
    generator = fast_chat_turn(sample_messages, chat_turn_dependencies)

    packets = []
    for packet in generator:
        packets.append(packet)

    # Debug: print all packets to understand what we're getting
    print(f"Tool call test - Total packets: {len(packets)}")
    for i, packet in enumerate(packets):
        print(f"Tool call test - Packet {i}: {packet}")

    # After cancellation during tool call, we should see MessageStart, SectionEnd, then OverallStop
    # The "Cancelled" MessageStart is shown when cancelling during tool calls/reasoning
    assert (
        len(packets) >= 3
    ), f"Expected at least 3 packets after tool call cancellation, got {len(packets)}"

    # The last packet should be OverallStop
    assert packets[-1].obj.type == "stop", "Last packet should be OverallStop"

    # The second-to-last should be SectionEnd
    assert (
        packets[-2].obj.type == "section_end"
    ), "Second-to-last packet should be SectionEnd"

    # The third-to-last should be MessageStart with "Cancelled" content
    assert (
        packets[-3].obj.type == "message_start"
    ), "Third-to-last packet should be MessageStart"
    assert isinstance(packets[-3].obj, MessageStart)
    assert (
        packets[-3].obj.content == "Cancelled"
    ), "MessageStart should contain 'Cancelled'"

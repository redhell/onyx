"""
Unit tests for fast_chat_turn functionality.

This module contains unit tests for the fast_chat_turn function, which handles
chat turn processing with agent-based interactions. The tests use dependency
injection with simple fake versions of all dependencies except for the emitter
(which is created by the unified_event_stream decorator) and dependencies_to_maybe_remove
(which should be passed in by the test writer).
"""

from uuid import uuid4

import pytest
from agents import FunctionTool
from agents import Model
from agents import ModelResponse
from agents import ModelSettings
from agents import ModelTracing
from agents import Usage
from agents.items import ResponseOutputMessage
from agents.items import ResponseOutputText
from openai.types.responses.response_usage import InputTokensDetails
from openai.types.responses.response_usage import OutputTokensDetails

from onyx.agents.agent_search.dr.enums import ResearchType
from onyx.chat.turn.models import ChatTurnDependencies
from onyx.chat.turn.models import DependenciesToMaybeRemove
from onyx.llm.interfaces import LLM
from onyx.llm.interfaces import LLMConfig
from onyx.server.query_and_chat.streaming_models import OverallStop
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageGenerationTool,
)
from onyx.tools.tool_implementations.okta_profile.okta_profile_tool import (
    OktaProfileTool,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool


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
        text_content = ResponseOutputText(text="fake response")
        message = ResponseOutputMessage(role="assistant", content=[text_content])

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

    async def stream_response(
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
    ):
        """Fake stream_response method that yields no events."""

        # Return an empty async iterator
        async def empty_iterator():
            if False:  # This ensures it's a proper async generator
                yield

        return empty_iterator()


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
        text_content = ResponseOutputText(text="fake response")
        message = ResponseOutputMessage(role="assistant", content=[text_content])

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

    async def stream_response(
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
    ):
        raise Exception("Fake exception")


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


class FakeSearchTool:
    """Simple fake SearchTool for testing."""

    def __init__(self):
        self.name = "search"
        self.call_count = 0

    def run(self, *args, **kwargs):
        self.call_count += 1
        return []


class FakeImageGenerationTool:
    """Simple fake ImageGenerationTool for testing."""

    def __init__(self):
        self.name = "image_generation"
        self.call_count = 0

    def run(self, *args, **kwargs):
        self.call_count += 1
        return []


class FakeOktaProfileTool:
    """Simple fake OktaProfileTool for testing."""

    def __init__(self):
        self.name = "okta_profile"
        self.call_count = 0

    def run(self, *args, **kwargs):
        self.call_count += 1
        return []


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
def fake_search_pipeline() -> SearchTool:
    """Fixture providing a fake search tool."""
    return FakeSearchTool()


@pytest.fixture
def fake_image_generation_tool() -> ImageGenerationTool:
    """Fixture providing a fake image generation tool."""
    return FakeImageGenerationTool()


@pytest.fixture
def fake_okta_profile_tool() -> OktaProfileTool:
    """Fixture providing a fake Okta profile tool."""
    return FakeOktaProfileTool()


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
    fake_search_pipeline: SearchTool,
    fake_image_generation_tool: ImageGenerationTool,
    fake_okta_profile_tool: OktaProfileTool,
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
        emitter=None,  # Set by unified_event_stream decorator
        search_pipeline=fake_search_pipeline,
        image_generation_tool=fake_image_generation_tool,
        okta_profile_tool=fake_okta_profile_tool,
        dependencies_to_maybe_remove=dependencies_to_maybe_remove,
    )


@pytest.fixture
def fake_failing_model() -> Model:
    return FakeFailingModel()


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
    """Test that demonstrates calling fast_chat_turn with dependency injection.

    This test shows how to actually call the fast_chat_turn function using
    the dependency injection setup. The function will run with fake implementations
    and the unified_event_stream decorator will handle the emitter creation.
    """
    # Import the function
    from onyx.chat.turn.fast_chat_turn import fast_chat_turn

    # Call the function - it returns a generator due to the unified_event_stream decorator
    generator = fast_chat_turn(sample_messages, chat_turn_dependencies)
    packets = list(generator)
    assert packets == [Packet(ind=0, obj=OverallStop(type="stop"))]


# TODO: Figure this one out
def test_fast_chat_turn_catch_exception(
    chat_turn_dependencies: ChatTurnDependencies,
    sample_messages: list[dict],
    fake_failing_model: Model,
):
    """Test that demonstrates calling fast_chat_turn with dependency injection.

    This test shows how to actually call the fast_chat_turn function using
    the dependency injection setup. The function will run with fake implementations
    and the unified_event_stream decorator will handle the emitter creation.
    """
    # Import the function
    from onyx.chat.turn.fast_chat_turn import fast_chat_turn

    chat_turn_dependencies.llm_model = fake_failing_model

    # Call the function - it returns a generator due to the unified_event_stream decorator
    generator = fast_chat_turn(sample_messages, chat_turn_dependencies)
    packets = list(generator)
    assert packets == [Packet(ind=0, obj=OverallStop(type="stop"))]

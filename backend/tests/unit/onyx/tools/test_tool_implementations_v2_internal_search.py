from uuid import uuid4

import pytest
from agents import RunContextWrapper

from onyx.agents.agent_search.dr.models import IterationAnswer
from onyx.agents.agent_search.dr.models import IterationInstructions
from onyx.chat.turn.models import ChatTurnContext
from onyx.chat.turn.models import DependenciesToMaybeRemove
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.context.search.models import InferenceSection
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.server.query_and_chat.streaming_models import SavedSearchDoc
from onyx.server.query_and_chat.streaming_models import SearchToolDelta
from onyx.server.query_and_chat.streaming_models import SearchToolStart
from onyx.server.query_and_chat.streaming_models import SectionEnd
from onyx.tools.tool_implementations.search.search_tool import (
    SEARCH_RESPONSE_SUMMARY_ID,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool


# =============================================================================
# Helper Functions and Base Classes for DRY Principles
# =============================================================================


class FakeEmitter:
    """Fake emitter for dependency injection"""

    def __init__(self):
        self.emitted_events = []

    def emit(self, packet: Packet):
        self.emitted_events.append(packet)


class FakeAggregatedContext:
    """Fake aggregated context for dependency injection"""

    def __init__(self):
        self.global_iteration_responses = []
        self.cited_documents = []


class FakeRunDependencies:
    """Fake run dependencies for dependency injection"""

    def __init__(self):
        self.emitter = FakeEmitter()
        self.dependencies_to_maybe_remove = None
        self.redis_client = None
        # Set up mock database session
        from unittest.mock import MagicMock

        self.db_session = MagicMock()
        # Configure the scalar method to return our mock tool
        mock_tool = FakeTool()
        self.db_session.scalar.return_value = mock_tool


class FakeTool:
    """Mock Tool object for testing"""

    def __init__(self, tool_id: int = 1, name: str = SearchTool.__name__):
        self.id = tool_id
        self.name = name


class FakeSearchPipeline:
    """Fake search pipeline for dependency injection"""

    def __init__(self, responses=None, should_raise_exception=False):
        self.responses = responses or []
        self.should_raise_exception = should_raise_exception
        self.run_called = False
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_called = True
        self.run_kwargs = kwargs
        if self.should_raise_exception:
            raise Exception("Test exception from search pipeline")
        return self.responses


def create_fake_inference_chunk(
    document_id="doc1", semantic_identifier="test_doc", blurb="Test content", chunk_id=0
) -> InferenceChunk:
    """Create a fake InferenceChunk for testing"""
    return InferenceChunk(
        document_id=document_id,
        chunk_id=chunk_id,
        source_type=DocumentSource.WEB,
        semantic_identifier=semantic_identifier,
        title=semantic_identifier,
        boost=1,
        recency_bias=1.0,
        score=0.95,
        hidden=False,
        is_relevant=True,
        relevance_explanation="Relevant to query",
        metadata={},
        match_highlights=[],
        doc_summary=blurb,
        chunk_context=blurb,
        blurb=blurb,
        content=blurb,  # Required by BaseChunk
        source_links=None,  # Required by BaseChunk
        image_file_id=None,  # Required by BaseChunk
        section_continuation=False,  # Required by BaseChunk
        updated_at=None,
        primary_owners=[],
        secondary_owners=[],
        large_chunk_reference_ids=[],
        is_federated=False,
    )


def create_fake_inference_section(
    document_id="doc1", semantic_identifier="test_doc", blurb="Test content"
) -> InferenceSection:
    """Create a fake InferenceSection for testing"""
    center_chunk = create_fake_inference_chunk(
        document_id=document_id,
        semantic_identifier=semantic_identifier,
        blurb=blurb,
    )
    return InferenceSection(
        center_chunk=center_chunk,
        chunks=[center_chunk],
        combined_content=blurb,
    )


class FakeSearchResponse:
    """Fake search response for testing"""

    def __init__(self, response_id, top_sections=None):
        self.id = response_id
        self.response = FakeSearchResponseSummary(top_sections or [])


class FakeSearchResponseSummary:
    """Fake search response summary for testing"""

    def __init__(self, top_sections):
        self.top_sections = top_sections


def create_fake_database_session():
    """Create a fake SQLAlchemy Session for testing"""
    from unittest.mock import Mock
    from sqlalchemy.orm import Session

    # Create a mock that behaves like a real Session
    fake_session = Mock(spec=Session)
    fake_session.committed = False
    fake_session.rolled_back = False

    def mock_commit():
        fake_session.committed = True

    def mock_rollback():
        fake_session.rolled_back = True

    fake_session.commit = mock_commit
    fake_session.rollback = mock_rollback
    fake_session.add = Mock()
    fake_session.flush = Mock()
    fake_session.query = Mock(return_value=FakeQuery())
    fake_session.execute = Mock(return_value=FakeResult())

    return fake_session


class FakeQuery:
    """Fake SQLAlchemy Query for testing"""

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return None

    def all(self):
        return []


class FakeResult:
    """Fake SQLAlchemy Result for testing"""

    def scalar(self):
        return None

    def fetchall(self):
        return []


class FakeSessionContextManager:
    """Fake session context manager for testing"""

    def __init__(self, session=None):
        self.session = session or create_fake_database_session()

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FakeRedis:
    """Fake Redis client for testing"""

    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

    def delete(self, key):
        return self.data.pop(key, 0)

    def exists(self, key):
        return 1 if key in self.data else 0


# =============================================================================
# Test Helper Functions
# =============================================================================


def create_fake_run_context(
    current_run_step: int = 0,
    dependencies_to_maybe_remove: DependenciesToMaybeRemove = None,
    redis_client: FakeRedis = None,
) -> RunContextWrapper[ChatTurnContext]:
    """Create a real RunContextWrapper with fake dependencies"""

    # Create fake dependencies
    emitter = FakeEmitter()
    aggregated_context = FakeAggregatedContext()

    run_dependencies = FakeRunDependencies()
    run_dependencies.emitter = emitter
    run_dependencies.dependencies_to_maybe_remove = dependencies_to_maybe_remove
    run_dependencies.redis_client = redis_client

    # Create the actual context object
    context = ChatTurnContext(
        current_run_step=current_run_step,
        iteration_instructions=[],
        aggregated_context=aggregated_context,
        run_dependencies=run_dependencies,
    )

    # Create the run context wrapper
    run_context = RunContextWrapper(context=context)

    return run_context


def create_fake_dependencies_to_maybe_remove() -> DependenciesToMaybeRemove:
    """Create fake dependencies to maybe remove"""
    return DependenciesToMaybeRemove(
        chat_session_id=uuid4(),
        message_id=123,
        research_type=None,  # Not needed for this test
    )


def create_fake_search_pipeline_with_results(
    sections=None, should_raise_exception=False
):
    """Create a fake search pipeline with test results"""
    if sections is None:
        sections = [
            create_fake_inference_section(
                document_id="doc1",
                semantic_identifier="test_doc_1",
                blurb="First test document content",
            ),
            create_fake_inference_section(
                document_id="doc2",
                semantic_identifier="test_doc_2",
                blurb="Second test document content",
            ),
        ]

    responses = [
        FakeSearchResponse(
            response_id=SEARCH_RESPONSE_SUMMARY_ID,
            top_sections=sections,
        ),
    ]

    return FakeSearchPipeline(
        responses=responses, should_raise_exception=should_raise_exception
    )


def create_fake_search_pipeline_empty():
    """Create a fake search pipeline with no results"""
    return FakeSearchPipeline(responses=[])


def create_fake_search_pipeline_multiple_responses():
    """Create a fake search pipeline with multiple responses"""
    test_sections = [create_fake_inference_section()]
    responses = [
        FakeSearchResponse(response_id="other_response_id", top_sections=[]),
        FakeSearchResponse(
            response_id=SEARCH_RESPONSE_SUMMARY_ID,
            top_sections=test_sections,
        ),
        FakeSearchResponse(response_id="another_response_id", top_sections=[]),
    ]
    return FakeSearchPipeline(responses=responses)


def run_internal_search_core_with_dependencies(
    run_context: RunContextWrapper[ChatTurnContext],
    query: str,
    search_pipeline: FakeSearchPipeline,
    session_context_manager: FakeSessionContextManager = None,
    redis_client: FakeRedis = None,
) -> list:
    """Helper function to run the real _internal_search_core with injected dependencies"""
    from unittest.mock import patch
    from onyx.tools.tool_implementations_v2.internal_search import _internal_search_core

    # Patch the dependencies that the real function uses
    with patch(
        "onyx.tools.tool_implementations_v2.internal_search.get_session_with_current_tenant"
    ) as mock_get_session, patch(
        "onyx.tools.tool_implementations_v2.internal_search.get_tool_by_name"
    ) as mock_get_tool_by_name:

        # Set up the session context manager mock
        if session_context_manager:
            mock_get_session.return_value = session_context_manager
        else:
            mock_get_session.return_value = FakeSessionContextManager()

        # Set up the get_tool_by_name mock to return our fake tool
        mock_get_tool_by_name.return_value = FakeTool()

        # Set up the Redis client in the run context if provided
        if redis_client:
            run_context.context.run_dependencies.redis_client = redis_client

        # Call the real _internal_search_core function
        return _internal_search_core(run_context, query, search_pipeline)


class FakeSearchToolOverrideKwargs:
    """Fake search tool override kwargs for testing"""

    def __init__(
        self,
        force_no_rerank=True,
        alternate_db_session=None,
        skip_query_analysis=True,
        original_query=None,
    ):
        self.force_no_rerank = force_no_rerank
        self.alternate_db_session = alternate_db_session
        self.skip_query_analysis = skip_query_analysis
        self.original_query = original_query


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def fake_emitter() -> FakeEmitter:
    """Fixture providing a fake emitter implementation."""
    return FakeEmitter()


@pytest.fixture
def fake_aggregated_context() -> FakeAggregatedContext:
    """Fixture providing a fake aggregated context implementation."""
    return FakeAggregatedContext()


@pytest.fixture
def fake_run_dependencies() -> FakeRunDependencies:
    """Fixture providing a fake run dependencies implementation."""
    return FakeRunDependencies()


@pytest.fixture
def fake_redis_client() -> FakeRedis:
    """Fixture providing a fake Redis client."""
    return FakeRedis()


@pytest.fixture
def fake_dependencies_to_maybe_remove() -> DependenciesToMaybeRemove:
    """Fixture providing fake dependencies to maybe remove."""
    return create_fake_dependencies_to_maybe_remove()


@pytest.fixture
def fake_run_context(
    fake_dependencies_to_maybe_remove: DependenciesToMaybeRemove,
    fake_redis_client: FakeRedis,
) -> RunContextWrapper[ChatTurnContext]:
    """Fixture providing a complete RunContextWrapper with fake implementations."""
    return create_fake_run_context(
        dependencies_to_maybe_remove=fake_dependencies_to_maybe_remove,
        redis_client=fake_redis_client,
    )


@pytest.fixture
def fake_search_pipeline() -> FakeSearchPipeline:
    """Fixture providing a fake search pipeline."""
    return create_fake_search_pipeline_with_results()


@pytest.fixture
def fake_session_context_manager() -> FakeSessionContextManager:
    """Fixture providing a fake session context manager."""
    return FakeSessionContextManager()


# =============================================================================
# Test Functions
# =============================================================================


def test_internal_search_core_basic_functionality(
    fake_run_context: RunContextWrapper[ChatTurnContext],
    fake_session_context_manager: FakeSessionContextManager,
):
    """Test basic functionality of _internal_search_core function using dependency injection"""
    # Arrange
    query = "test search query"
    test_pipeline = create_fake_search_pipeline_with_results()

    # Act
    result = run_internal_search_core_with_dependencies(
        fake_run_context, query, test_pipeline, fake_session_context_manager
    )

    # Assert
    assert isinstance(result, list)
    assert len(result) == 2

    # Verify context was updated (decorator increments current_run_step)
    assert fake_run_context.context.current_run_step == 2
    assert len(fake_run_context.context.iteration_instructions) == 1
    assert (
        len(fake_run_context.context.aggregated_context.global_iteration_responses) == 1
    )
    # Verify cited_documents were added to aggregated_context
    assert len(fake_run_context.context.aggregated_context.cited_documents) == 2
    assert (
        fake_run_context.context.aggregated_context.cited_documents[
            0
        ].center_chunk.document_id
        == "doc1"
    )
    assert (
        fake_run_context.context.aggregated_context.cited_documents[
            1
        ].center_chunk.document_id
        == "doc2"
    )

    # Check iteration instruction
    instruction = fake_run_context.context.iteration_instructions[0]
    assert isinstance(instruction, IterationInstructions)
    assert instruction.iteration_nr == 1
    assert instruction.purpose == "Searching internally for information"
    assert (
        f"I am now using Internal Search to gather information on {query}"
        in instruction.reasoning
    )

    # Check iteration answer
    answer = fake_run_context.context.aggregated_context.global_iteration_responses[0]
    assert isinstance(answer, IterationAnswer)
    assert answer.tool == SearchTool.__name__
    assert answer.tool_id == 1
    assert answer.iteration_nr == 1
    assert answer.question == query
    assert (
        answer.reasoning
        == f"I am now using Internal Search to gather information on {query}"
    )
    assert answer.answer == "Cool"
    assert len(answer.cited_documents) == 2

    # Verify emitter events were captured
    emitter = fake_run_context.context.run_dependencies.emitter
    assert len(emitter.emitted_events) == 4

    # Check the types of emitted events
    assert isinstance(emitter.emitted_events[0].obj, SearchToolStart)
    assert isinstance(emitter.emitted_events[1].obj, SearchToolDelta)
    assert isinstance(emitter.emitted_events[2].obj, SearchToolDelta)
    assert isinstance(emitter.emitted_events[3].obj, SectionEnd)

    # Check the first SearchToolDelta (query)
    first_delta = emitter.emitted_events[1].obj
    assert first_delta.queries == [query]
    assert first_delta.documents is None

    # Check the second SearchToolDelta (documents)
    second_delta = emitter.emitted_events[2].obj
    assert second_delta.queries is None
    assert second_delta.documents is not None
    assert len(second_delta.documents) == 2

    # Verify the SavedSearchDoc objects
    first_doc = second_delta.documents[0]
    assert isinstance(first_doc, SavedSearchDoc)
    assert first_doc.document_id == "doc1"
    assert first_doc.semantic_identifier == "test_doc_1"
    assert first_doc.blurb == "First test document content"
    assert first_doc.source_type == DocumentSource.WEB
    assert first_doc.is_internet is False

    # Verify the pipeline was called with correct parameters
    assert test_pipeline.run_called
    assert test_pipeline.run_kwargs["query"] == query
    assert test_pipeline.run_kwargs["override_kwargs"].force_no_rerank is True
    assert test_pipeline.run_kwargs["override_kwargs"].skip_query_analysis is True
    assert test_pipeline.run_kwargs["override_kwargs"].original_query == query

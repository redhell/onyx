from dataclasses import dataclass
from uuid import UUID

from agents import FunctionTool
from agents import Model
from redis.client import Redis
from sqlalchemy.orm import Session

from onyx.agents.agent_search.dr.enums import ResearchType
from onyx.agents.agent_search.dr.models import AggregatedDRContext
from onyx.agents.agent_search.dr.models import IterationInstructions
from onyx.chat.turn.infra.chat_turn_event_stream import Emitter
from onyx.tools.tool_implementations.search.search_tool import SearchTool


@dataclass
class DependenciesToMaybeRemove:
    chat_session_id: UUID
    message_id: int
    research_type: ResearchType


@dataclass
class ChatTurnDependencies:
    llm_model: Model
    db_session: Session
    tools: list[FunctionTool]
    redis_client: Redis | None = None
    emitter: Emitter | None = None
    search_pipeline: SearchTool | None = None
    dependencies_to_maybe_remove: DependenciesToMaybeRemove | None = None


@dataclass
class MyContext:
    """Context class to hold search tool and other dependencies"""

    run_dependencies: ChatTurnDependencies | None = None
    needs_compaction: bool = False
    current_run_step: int = 0
    # TODO: Figure out a cleaner way to persist information.
    aggregated_context: AggregatedDRContext | None = None
    iteration_instructions: list[IterationInstructions] | None = None
    web_fetch_results: list[dict] | None = None

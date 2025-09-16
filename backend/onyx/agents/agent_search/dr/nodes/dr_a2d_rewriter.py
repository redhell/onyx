from datetime import datetime
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from onyx.agents.agent_search.dr.states import FinalUpdate
from onyx.agents.agent_search.dr.states import MainState
from onyx.agents.agent_search.dr.states import OrchestrationUpdate
from onyx.agents.agent_search.models import GraphConfig
from onyx.agents.agent_search.shared_graph_utils.utils import (
    get_langgraph_node_log_string,
)
from onyx.utils.logger import setup_logger


logger = setup_logger()

_SOURCE_MATERIAL_PROMPT = "Can yut please put together all of the supporting material?"


def rewriter(
    state: MainState, config: RunnableConfig, writer: StreamWriter = lambda _: None
) -> FinalUpdate | OrchestrationUpdate:
    """
    LangGraph node to close the DR process and finalize the answer.
    """

    node_start_time = datetime.now()
    # TODO: generate final answer using all the previous steps
    # (right now, answers from each step are concatenated onto each other)
    # Also, add missing fields once usage in UI is clear.

    state.current_step_nr

    graph_config = cast(GraphConfig, config["metadata"]["config"])
    base_question = state.original_question
    if not base_question:
        raise ValueError("Question is required for closer")

    graph_config.behavior.research_type

    state.assistant_system_prompt
    state.assistant_task_prompt

    final_answer = state.final_answer

    all_cited_documents = state.all_cited_documents

    # iteration_responses = state.iteration_responses

    claims = []

    # for iteration_response in iteration_responses:
    #     claims.extend(iteration_response.claims)

    claims = list(set(claims))

    # aggregated_context = aggregate_context(iteration_responses, include_documents=True)

    # aggregated_context_wo_docs = aggregate_context(
    #     iteration_responses, include_documents=False
    # )

    return FinalUpdate(
        final_answer=final_answer,
        all_cited_documents=all_cited_documents,
        log_messages=[
            get_langgraph_node_log_string(
                graph_component="main",
                node_name="closer",
                node_start_time=node_start_time,
            )
        ],
    )

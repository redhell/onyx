from typing import cast

from agents import Agent
from agents import ModelSettings
from agents import RunItemStreamEvent
from agents.extensions.models.litellm_model import LitellmModel

from onyx.agents.agent_search.dr.models import AggregatedDRContext
from onyx.chat.stop_signal_checker import is_connected
from onyx.chat.turn.infra.chat_turn_event_stream import OnyxRunner
from onyx.chat.turn.infra.chat_turn_orchestration import unified_event_stream
from onyx.chat.turn.infra.packet_translation import default_packet_translation
from onyx.chat.turn.infra.session_sink import save_iteration
from onyx.chat.turn.models import MyContext
from onyx.chat.turn.models import RunDependencies
from onyx.server.query_and_chat.streaming_models import OverallStop
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.server.query_and_chat.streaming_models import SectionEnd
from onyx.tools.tool_implementations_v2.internal_search import internal_search
from onyx.tools.tool_implementations_v2.web import web_fetch
from onyx.tools.tool_implementations_v2.web import web_search


# TODO: Dependency injection?
@unified_event_stream
def fast_chat_turn(messages: list[dict], dependencies: RunDependencies) -> None:
    ctx = MyContext(
        run_dependencies=dependencies,
        aggregated_context=AggregatedDRContext(
            context="context",
            cited_documents=[],
            is_internet_marker_dict={},
            global_iteration_responses=[],  # TODO: the only field that matters for now
        ),
        iteration_instructions=[],
    )
    agent = Agent(
        name="Assistant",
        model=LitellmModel(
            model=dependencies.llm.config.model_name,
            api_key=dependencies.llm.config.api_key,
        ),
        tools=[web_search, web_fetch, internal_search],
        model_settings=ModelSettings(
            temperature=0.0,
            include_usage=True,
        ),
    )

    bridge = OnyxRunner().run_streamed(agent, messages, context=ctx, max_turns=100)
    final_answer = "filler final answer"
    for ev in bridge.events():
        # TODO: Wrap in some cancellation handler
        if not is_connected(
            dependencies.dependencies_to_maybe_remove.chat_session_id,
            dependencies.redis_client,
        ):
            dependencies.emitter.emit(
                Packet(ind=ctx.current_run_step, obj=SectionEnd(type="section_end"))
            )
            dependencies.emitter.emit(
                Packet(ind=ctx.current_run_step, obj=OverallStop(type="stop"))
            )
            bridge.cancel()
            break
        ctx.current_run_step
        obj = default_packet_translation(ev)
        # TODO this obviously won't work for cancellation
        if isinstance(ev, RunItemStreamEvent):
            ev = cast(RunItemStreamEvent, ev)
            if ev.name == "message_output_created":
                final_answer = ev.item.raw_item.content[0].text
        if obj:
            dependencies.emitter.emit(Packet(ind=ctx.current_run_step, obj=obj))

    save_iteration(
        db_session=dependencies.db_session,
        message_id=dependencies.dependencies_to_maybe_remove.message_id,
        chat_session_id=dependencies.dependencies_to_maybe_remove.chat_session_id,
        research_type=dependencies.dependencies_to_maybe_remove.research_type,
        ctx=ctx,
        final_answer=final_answer,
        all_cited_documents=[],
    )
    # TODO: Error handling
    # Should there be a timeout and some error on the queue?
    dependencies.emitter.emit(
        Packet(ind=ctx.current_run_step, obj=OverallStop(type="stop"))
    )

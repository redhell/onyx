from queue import Queue

from agents import Agent
from agents import ModelSettings
from agents import RawResponsesStreamEvent

from onyx.agents.agent_search.dr.models import AggregatedDRContext
from onyx.chat.stop_signal_checker import is_connected
from onyx.chat.stop_signal_checker import reset_cancel_status
from onyx.chat.turn.infra.chat_turn_event_stream import unified_event_stream
from onyx.chat.turn.infra.session_sink import extract_final_answer_from_packets
from onyx.chat.turn.infra.session_sink import save_iteration
from onyx.chat.turn.infra.sync_agent_stream_adapter import SyncAgentStream
from onyx.chat.turn.models import ChatTurnContext
from onyx.chat.turn.models import ChatTurnDependencies
from onyx.server.query_and_chat.streaming_models import MessageDelta
from onyx.server.query_and_chat.streaming_models import MessageStart
from onyx.server.query_and_chat.streaming_models import OverallStop
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.server.query_and_chat.streaming_models import PacketObj
from onyx.server.query_and_chat.streaming_models import SectionEnd
from onyx.tools.tool_implementations_v2.reasoning import reasoning_tool


@unified_event_stream
def fast_chat_turn(messages: list[dict], dependencies: ChatTurnDependencies) -> None:
    reset_cancel_status(
        dependencies.dependencies_to_maybe_remove.chat_session_id,
        dependencies.redis_client,
    )
    ctx = ChatTurnContext(
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
        model=dependencies.llm_model,
        tools=dependencies.tools + [reasoning_tool],
        model_settings=ModelSettings(
            temperature=0.0,
            include_usage=True,
        ),
    )
    agent_stream = SyncAgentStream(
        agent=agent,
        input=messages,
        context=ctx,
        max_turns=100,  # TODO: magic number
    )
    for ev in agent_stream:
        connected = is_connected(
            dependencies.dependencies_to_maybe_remove.chat_session_id,
            dependencies.redis_client,
        )
        if not connected:
            _emit_clean_up_packets(dependencies, ctx)
            agent_stream.cancel()
            break
        ctx.current_run_step
        obj = default_packet_translation(ev)
        if obj:
            dependencies.emitter.emit(Packet(ind=ctx.current_run_step, obj=obj))
    save_iteration(
        db_session=dependencies.db_session,
        message_id=dependencies.dependencies_to_maybe_remove.message_id,
        chat_session_id=dependencies.dependencies_to_maybe_remove.chat_session_id,
        research_type=dependencies.dependencies_to_maybe_remove.research_type,
        ctx=ctx,
        final_answer=extract_final_answer_from_packets(
            dependencies.emitter.packet_history
        ),
        all_cited_documents=[],
    )
    dependencies.emitter.emit(
        Packet(ind=ctx.current_run_step, obj=OverallStop(type="stop"))
    )


# TODO: Maybe in general there's a cleaner way to handle cancellation in the middle of a tool call?
def _emit_clean_up_packets(
    dependencies: ChatTurnDependencies, ctx: ChatTurnContext
) -> None:
    if not (
        dependencies.emitter.packet_history
        and dependencies.emitter.packet_history[-1].obj.type == "message_delta"
    ):
        dependencies.emitter.emit(
            Packet(
                ind=ctx.current_run_step,
                obj=MessageStart(
                    type="message_start", content="Cancelled", final_documents=None
                ),
            )
        )
    dependencies.emitter.emit(
        Packet(ind=ctx.current_run_step, obj=SectionEnd(type="section_end"))
    )


class Emitter:
    """Use this inside tools to emit arbitrary UI progress."""

    def __init__(self, bus: Queue):
        self.bus = bus
        self.packet_history: list[Packet] = []

    def emit(self, packet: Packet) -> None:
        self.bus.put(packet)
        self.packet_history.append(packet)


def default_packet_translation(ev: object) -> PacketObj | None:
    if isinstance(ev, RawResponsesStreamEvent):
        # TODO: might need some variation here for different types of models
        # OpenAI packet translator
        obj: PacketObj | None = None
        if ev.data.type == "response.content_part.added":
            obj = MessageStart(type="message_start", content="", final_documents=None)
        elif ev.data.type == "response.output_text.delta":
            obj = MessageDelta(type="message_delta", content=ev.data.delta)
        elif ev.data.type == "response.content_part.done":
            obj = SectionEnd(type="section_end")
        return obj
    return None

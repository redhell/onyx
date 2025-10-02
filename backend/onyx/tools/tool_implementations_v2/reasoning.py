from agents import Agent
from agents import function_tool
from agents import ModelSettings
from agents import RunContextWrapper

from onyx.chat.stop_signal_checker import is_connected
from onyx.chat.turn.infra.chat_turn_event_stream import OnyxRunner
from onyx.chat.turn.infra.packet_translation import default_packet_translation
from onyx.chat.turn.models import ChatTurnContext
from onyx.chat.turn.models import ChatTurnDependencies
from onyx.server.query_and_chat.streaming_models import MessageStart
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.server.query_and_chat.streaming_models import SectionEnd
from onyx.utils.threadpool_concurrency import wait_on_background


@function_tool
def reasoning_tool(run_context: RunContextWrapper[ChatTurnContext], query: str) -> str:
    """
    Reason about the query and return the answer.
    """
    agent = Agent(
        name="Assistant",
        model=run_context.context.run_dependencies.llm_model,
        tools=[],
        model_settings=ModelSettings(
            temperature=0.0,
            include_usage=True,
        ),
    )
    bridge, thread = OnyxRunner().run_streamed_in_background(
        agent,
        [
            {"role": "user", "content": query},
        ],
        context=run_context.context,
    )
    for ev in bridge.events():
        if not is_connected(
            run_context.context.dependencies_to_maybe_remove.chat_session_id,
            run_context.context.redis_client,
        ):
            _emit_clean_up_packets(run_context.context, run_context.context)
            bridge.cancel()
            break
        run_context.context.current_run_step
        obj = default_packet_translation(ev)
        if obj:
            run_context.context.emitter.emit(
                Packet(ind=run_context.context.current_run_step, obj=obj)
            )
    wait_on_background(thread)
    return "Reasoning tool"


def _emit_clean_up_packets(
    dependencies: ChatTurnDependencies, ctx: ChatTurnContext
) -> None:
    # Tool call / reasoning cancelled
    if (
        dependencies.emitter.packet_history
        and dependencies.emitter.packet_history[-1].obj.type != "message_delta"
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

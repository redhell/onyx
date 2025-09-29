import contextvars
import threading
from collections.abc import Callable
from collections.abc import Generator
from queue import Queue
from typing import Any
from typing import Dict
from typing import List

from onyx.chat.turn.infra.chat_turn_event_stream import Emitter
from onyx.chat.turn.models import RunDependencies
from onyx.server.query_and_chat.streaming_models import OverallStop
from onyx.server.query_and_chat.streaming_models import Packet
from onyx.server.query_and_chat.streaming_models import PacketException


def unified_event_stream(
    turn_func: Callable[[List[Dict[str, Any]], RunDependencies], None],
) -> Callable[[List[Dict[str, Any]], RunDependencies], Generator[Packet, None]]:
    """
    Decorator that wraps a turn_func to provide event streaming capabilities.

    Usage:
    @unified_event_stream
    def my_turn_func(messages, dependencies):
        # Your turn logic here
        pass

    Then call it like:
    generator = my_turn_func(messages, dependencies)
    """

    def wrapper(
        messages: List[Dict[str, Any]], dependencies: RunDependencies
    ) -> Generator[Packet, None]:
        bus: Queue = Queue()
        emitter = Emitter(bus)
        current_context = contextvars.copy_context()
        dependencies.emitter = emitter

        def run_with_exception_capture():
            try:
                current_context.run(turn_func, messages, dependencies)
            except Exception as e:
                emitter.emit(
                    Packet(ind=0, obj=PacketException(type="error", exception=e))
                )

        t = threading.Thread(target=run_with_exception_capture, daemon=True)
        t.start()

        while True:
            pkt: Packet = emitter.bus.get()
            if pkt.obj == OverallStop(type="stop"):
                yield pkt
                break
            elif isinstance(pkt.obj, PacketException):
                raise pkt.obj.exception
            else:
                yield pkt

    return wrapper

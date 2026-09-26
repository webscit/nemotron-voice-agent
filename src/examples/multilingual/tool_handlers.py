# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Server-side handlers for tools that actually run on the client.

RTVI (see Pipecat's ``RTVIProcessor``/``RTVIObserver``) already carries a
function call end-to-end without any custom messaging:

1. The LLM calls a registered tool as normal. ``NvidiaLLMService`` broadcasts
   ``FunctionCallInProgressFrame`` before invoking the handler below.
2. With ``RTVIObserverParams.function_call_report_level`` set to ``FULL`` for
   that tool name, the observer automatically turns that frame into an
   ``llm-function-call-in-progress`` message carrying the function name and
   arguments. The Pipecat client SDK's ``registerFunctionCallHandler`` reacts
   to that event, runs the tool locally, and replies with an
   ``llm-function-call-result`` message on its own.
3. ``RTVIProcessor`` turns that reply directly into a ``FunctionCallResultFrame``
   pushed into the pipeline (``RTVIProcessor._handle_function_call_result``),
   which the assistant context aggregator matches by ``tool_call_id`` and
   folds into the conversation — regardless of what the handler below does.

So the handler registered here never computes a result itself: it only holds
the call open and applies a server-side timeout so a client that never
answers can't stall the conversation forever. ``ClientToolResultBridge`` is
the piece that lets the handler notice the client's answer *as it flows past*
so it can stop waiting instead of always sleeping the full timeout — without
it, every successful client answer would still be followed by a spurious
second ``FunctionCallResultFrame`` (and a second ``llm-function-call-stopped``
event on the client) once the timeout elapsed, since the handler would have
no way to know the call was already finished.

Registered with ``cancel_on_interruption=True`` (the default "sync" tool
pattern), not ``False``. Pipecat has a separate "async tool" pattern
(``cancel_on_interruption=False``, used for genuinely long-running background
work elsewhere in this repo) that lets the conversation continue while a tool
runs, at the cost of injecting the eventual result as a ``role="developer"``
message (see ``pipecat.processors.aggregators.async_tool_messages``) instead
of updating the normal ``role="tool"`` message in place. That extra message
shape is a convention some LLMs are trained to recognize and others are not —
Nemotron 3.5 Lightning (via vLLM here) does not: it saw the ``developer``
message land in context and kept telling the user the tool was "still
running," even though the real result was sitting right there. A client
round-trip is fast enough that the plain sync pattern (cancelled outright if
the user interrupts, which is the right behavior for a quick lookup anyway)
is both simpler and actually understood by the model.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.frames.frames import Frame, FunctionCallResultFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams


class ClientToolResultBridge(FrameProcessor):
    """Pass-through processor that watches for client-delivered tool results.

    ``RTVIProcessor`` pushes the client's ``FunctionCallResultFrame`` straight
    into the pipeline (see module docstring) with no event our own code can
    subscribe to. This processor sits later in the pipeline, and the instant
    that frame passes through it resolves a future for the matching
    ``tool_call_id`` — so ``build_client_tool_handler``'s wait ends the moment
    the real answer arrives, rather than always running the full timeout.

    Every frame is forwarded unchanged; this never consumes or alters
    anything, so the assistant context aggregator downstream still sees and
    processes the ``FunctionCallResultFrame`` normally.
    """

    def __init__(self):
        """Create the bridge with no pending waits yet."""
        super().__init__()
        self._pending: dict[str, asyncio.Future] = {}

    def wait_for_result(self, tool_call_id: str) -> asyncio.Future:
        """Return a future that resolves once ``tool_call_id``'s result passes through."""
        future = asyncio.get_running_loop().create_future()
        self._pending[tool_call_id] = future
        return future

    def cancel_wait(self, tool_call_id: str) -> None:
        """Stop tracking ``tool_call_id`` (call this once a wait is no longer needed)."""
        self._pending.pop(tool_call_id, None)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Resolve any pending wait for a matching result, then forward the frame."""
        await super().process_frame(frame, direction)
        if isinstance(frame, FunctionCallResultFrame):
            future = self._pending.pop(frame.tool_call_id, None)
            if future is not None and not future.done():
                future.set_result(None)
        await self.push_frame(frame, direction)


def build_client_tool_handler(timeout_secs: float, bridge: ClientToolResultBridge):
    """Return a handler that waits for the client to answer a forwarded tool call.

    The handler itself never produces a result on the happy path — the
    client's ``llm-function-call-result`` reply completes the call directly
    (see module docstring), and ``bridge`` lets this wait end as soon as that
    happens. Only if the client does not answer within ``timeout_secs`` does
    the handler finalize the call itself with an error, so the LLM can
    continue instead of waiting indefinitely.
    """

    async def handle_client_tool_timeout(params: FunctionCallParams) -> None:
        future = bridge.wait_for_result(params.tool_call_id)
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=timeout_secs)
            # The client's real result already flowed through the pipeline and
            # was applied directly by RTVIProcessor — nothing left to do here.
        except TimeoutError:
            bridge.cancel_wait(params.tool_call_id)
            logger.warning(
                f"Client did not answer '{params.function_name}' [{params.tool_call_id}] within {timeout_secs:.1f}s"
            )
            await params.result_callback({"error": "Timed out waiting for the client to answer this request."})

    return handle_client_tool_timeout

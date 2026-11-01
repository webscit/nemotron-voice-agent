# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call client tools chosen by a decision model after each LLM answer.

``DecisionToolsProcessor`` sits right after the LLM. When an answer's text is
complete, it asks the decision model (see ``client.py``) which value each
decision tool should take for the user's last message and that answer, and
sends the chosen calls to the client while TTS speaks the answer.

The calls stay out of the conversation: they are sent as RTVI
``llm-function-call-in-progress`` messages directly (the client runs them like
any delegated LLM tool call), and the client's ``FunctionCallResultFrame``
replies, recognized by their ``tool_call_id`` prefix, are dropped here before
the assistant aggregator could fold them into the context. The LLM never sees
these tools, so a call it did not make would only confuse it.

Replies spoken by the intent engine are skipped: they did not come from the
LLM, and an intent may already be the tool itself ("show me you are happy").
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.models import (
    LLMFunctionCallInProgressMessage,
    LLMFunctionCallInProgressMessageData,
)

from examples.shared.decision_tools.client import Decision, DecisionError, SystemOneClient
from examples.shared.decision_tools.config import DecisionTool
from examples.shared.intent_engine.matcher import strip_tags
from examples.shared.intent_engine.processor import INTENT_REPLY_METADATA

if TYPE_CHECKING:
    import asyncio

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pydantic import BaseModel

    from monitoring.recorder import SessionRecorder

# Tool calls sent by this processor; their client results never reach the context.
DECISION_CALL_PREFIX = "decision-"
EVENT_KIND = "decision_tool"
# Loading the model after an Ollama restart takes seconds (4-7 s on Jetson Thor, up to
# 14 s when its file is not in the page cache): the warm-up decision at session start waits that long, a turn does not.
WARM_UP_TIMEOUT_SECS = 120.0
WARM_UP_STATE = "User: Hello\nAssistant: Hello!"
_PROCESSOR_NAME = "DecisionTools"


def last_user_text(context: LLMContext) -> str:
    """Return the text of the most recent user message, speaker tags removed."""
    for message in reversed(context.get_messages()):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
            )
        return strip_tags(content) if isinstance(content, str) else ""
    return ""


def build_state(user_text: str, answer: str) -> str:
    """Return the exchange the decision model judges."""
    lines = [f"User: {user_text}"] if user_text else []
    lines.append(f"Assistant: {' '.join(answer.split())}")
    return "\n".join(lines)


class DecisionToolsProcessor(FrameProcessor):
    """Ask the decision model which client tools to call once each LLM answer is complete."""

    def __init__(
        self,
        *,
        context: LLMContext,
        tools: tuple[DecisionTool, ...],
        client: SystemOneClient,
        send_message: Callable[[BaseModel], Awaitable[None]],
        recorder: SessionRecorder | None = None,
    ):
        """Build the processor.

        ``context`` is the session's shared context, read for the user's last
        message. ``send_message`` delivers an RTVI message to the client.
        """
        super().__init__()
        self._context = context
        self._tools = {tool.name: tool for tool in tools}
        self._client = client
        self._send_message = send_message
        self._recorder = recorder
        self._answer: list[str] | None = None
        self._task: asyncio.Task | None = None
        self._warm_up_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Collect each LLM answer, decide on it once complete, and drop decision-call results."""
        await super().process_frame(frame, direction)
        if isinstance(frame, FunctionCallResultFrame) and frame.tool_call_id.startswith(DECISION_CALL_PREFIX):
            logger.debug(f"Decision tools: {frame.function_name} -> {frame.result!r}")
            return
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._warm_up_task = self.create_task(self._warm_up(), "warm-up")
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            self._answer = None if frame.metadata.get(INTENT_REPLY_METADATA) else []
        elif isinstance(frame, LLMTextFrame) and self._answer is not None:
            self._answer.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame) and self._answer is not None:
            answer, self._answer = "".join(self._answer).strip(), None
            if answer:
                await self._cancel()
                self._task = self.create_task(self._decide(answer), "decide")
        elif isinstance(frame, (InterruptionFrame, EndFrame, CancelFrame)):
            self._answer = None
            await self._cancel()
            if not isinstance(frame, InterruptionFrame) and self._warm_up_task and not self._warm_up_task.done():
                await self.cancel_task(self._warm_up_task)
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        """Release the decision model HTTP client."""
        await super().cleanup()
        await self._client.close()

    async def _cancel(self) -> None:
        if self._task and not self._task.done():
            await self.cancel_task(self._task)
        self._task = None

    async def _warm_up(self) -> None:
        """Load the decision model now, so the first answer does not wait for it."""
        started = time.monotonic()
        try:
            await self._client.decide(WARM_UP_STATE, tuple(self._tools.values()), timeout_secs=WARM_UP_TIMEOUT_SECS)
            logger.info(f"Decision tools: decision model ready ({_ms_since(started)} ms)")
        except DecisionError as exc:
            logger.warning(f"Decision tools: decision model warm-up failed ({exc})")

    async def _decide(self, answer: str) -> None:
        """Ask the decision model about ``answer`` and send the chosen calls."""
        started = time.monotonic()
        try:
            decisions = await self._client.decide(
                build_state(last_user_text(self._context), answer), tuple(self._tools.values())
            )
        except DecisionError as exc:
            logger.warning(f"Decision tools: decision model failed ({exc}); skipping this answer")
            self._record(outcome="error", error=str(exc), latency_ms=_ms_since(started))
            return
        latency_ms = _ms_since(started)
        for decision in decisions:
            tool = self._tools[decision.tool]
            outcome = self._outcome(tool, decision)
            logger.info(
                f"Decision tools: {tool.name} -> {decision.choice} "
                f"(confidence {decision.confidence:.2f}, {latency_ms} ms, {outcome})"
            )
            self._record(
                outcome=outcome,
                tool=tool.name,
                choice=decision.choice,
                confidence=decision.confidence,
                probabilities=decision.probabilities,
                latency_ms=latency_ms,
            )
            if outcome == "called":
                await self._call(tool, decision.choice)

    @staticmethod
    def _outcome(tool: DecisionTool, decision: Decision) -> str:
        if decision.choice not in tool.criteria:
            return "none"
        if decision.confidence < tool.min_confidence:
            return "low_confidence"
        return "called"

    async def _call(self, tool: DecisionTool, choice: str) -> None:
        """Send one tool call to the client, outside the LLM's function-call flow."""
        data = LLMFunctionCallInProgressMessageData(
            tool_call_id=f"{DECISION_CALL_PREFIX}{uuid.uuid4().hex[:16]}",
            function_name=tool.name,
            arguments=tool.call_arguments(choice),
        )
        await self._send_message(LLMFunctionCallInProgressMessage(data=data))

    def _record(self, **data: Any) -> None:
        if self._recorder:
            self._recorder.event(EVENT_KIND, processor=_PROCESSOR_NAME, **data)


def _ms_since(started: float) -> int:
    return round((time.monotonic() - started) * 1000)

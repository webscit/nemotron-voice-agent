# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline processor that answers matched device commands without the LLM.

Place it right before the LLM service. When the user aggregator emits the
context frame for a finished user turn, the processor matches the transcript:

- Miss: the frame goes on to the LLM unchanged.
- Hit: the frame is swallowed, a target executes the intent, and the processor
  emits the frames an LLM reply would (``LLMFullResponseStartFrame``, one
  ``LLMTextFrame``, ``LLMFullResponseEndFrame``). TTS, the assistant aggregator,
  chat-history summarization, and interruptions therefore behave as usual, and
  the reply lands in the context as an assistant message.

The processor always waits for the target's answer before speaking. If the
target cannot answer, the swallowed frame is released to the LLM.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.shared.intent_engine.config import ALLOWED_STATE_RESPONSE_KEYS, IntentEngineConfig
from examples.shared.intent_engine.matcher import IntentMatch, IntentMatcher, base_language
from examples.shared.intent_engine.targets import (
    ClientToolMap,
    ClientToolTarget,
    HomeAssistantTarget,
    IntentRequest,
    TargetUnavailable,
)

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.llm_service import LLMService

EVENT_KIND = "intent_engine"
METRIC_MATCH_SECS = "intent_match_secs"
METRIC_TARGET_SECS = "intent_target_secs"
_PROCESSOR_NAME = "IntentEngine"
_SYNC_RETRY_SECS = 30.0


class TurnTelemetry(Protocol):
    """The two ``SessionRecorder`` hooks the engine reports through."""

    def event(self, kind: str, *, processor: str | None = None, **data: Any) -> None:
        """Queue a timeline event for the current turn."""

    def metric(self, name: str, value: float, *, processor: str | None = None, model: str | None = None) -> None:
        """Queue a numeric metric sample for the current turn."""


class IntentEngineProcessor(FrameProcessor):
    """Match finished user turns against intents and execute hits directly."""

    def __init__(
        self,
        *,
        config: IntentEngineConfig,
        language: str,
        matcher: IntentMatcher,
        context: LLMContext,
        home_assistant: HomeAssistantTarget | None = None,
        client_tools: ClientToolTarget | None = None,
        tool_map: ClientToolMap | None = None,
        recorder: TurnTelemetry | None = None,
    ):
        """Build the processor.

        ``context`` is the session's shared context. The user text is read from
        it rather than from the incoming frame, because ``PerTurnReminderProcessor``
        forwards a copy whose last user message carries the language reminder.
        """
        super().__init__()
        self._config = config
        self._language = base_language(language)
        self._matcher = matcher
        self._context = context
        self._home_assistant = home_assistant
        self._client_tools = client_tools
        self._tool_map = tool_map or ClientToolMap()
        self._recorder = recorder
        self._turn_task: asyncio.Task | None = None
        self._sync_task: asyncio.Task | None = None
        self._last_sync_attempt = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Intercept finished user turns; forward everything else."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._start_sync()
            return
        if isinstance(frame, (InterruptionFrame, EndFrame, CancelFrame)):
            await self._cancel_turn()
        elif isinstance(frame, FunctionCallResultFrame) and self._client_tools:
            self._client_tools.observe_result(frame, direction)
        elif (
            isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
            and self._handle_user_turn(frame)
        ):
            return
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        """Release the Home Assistant HTTP client."""
        await super().cleanup()
        if self._home_assistant:
            await self._home_assistant.aclose()

    # ------------------------------------------------------------ entity sync
    def _start_sync(self) -> None:
        """Load Home Assistant names in the background (retried after a failure)."""
        if not self._home_assistant or self._home_assistant.slot_lists is not None:
            return
        if self._sync_task and not self._sync_task.done():
            return
        if self._last_sync_attempt and time.monotonic() - self._last_sync_attempt < _SYNC_RETRY_SECS:
            return
        self._last_sync_attempt = time.monotonic()
        self._sync_task = self.create_task(self._sync())

    async def _sync(self) -> None:
        try:
            lists = await self._home_assistant.sync()
            logger.info(f"Intent engine: synced {len(lists['name'].values)} Home Assistant entity names")
        except TargetUnavailable as exc:
            logger.warning(f"Intent engine: Home Assistant sync failed ({exc}); its intents stay inactive")

    # ------------------------------------------------------------- user turns
    def _user_text(self) -> str:
        """Return the text of the turn that just ended, or "" when the last message is not the user's."""
        messages = self._context.get_messages()
        last = messages[-1] if messages else None
        if not isinstance(last, dict) or last.get("role") != "user":
            return ""
        content = last.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
            return " ".join(part for part in parts if part)
        return ""

    def _route(self, match: IntentMatch) -> tuple[HomeAssistantTarget | ClientToolTarget | None, str]:
        """Pick the target for a match, or the reason the LLM must handle it."""
        if match.name in self._tool_map:
            if self._tool_map.defers_to_llm(match.name):
                return None, "deferred_to_llm"
            return self._client_tools, "" if self._client_tools else "target_not_ready"
        if self._home_assistant is None or self._home_assistant.slot_lists is None:
            return None, "target_not_ready"
        if match.name not in self._config.allowed_intents:
            return None, "not_allowed"
        if match.name == "HassGetState" and match.response_key not in ALLOWED_STATE_RESPONSE_KEYS:
            return None, "state_question"
        domains = match.slots.get("domain")
        domains = domains if isinstance(domains, list) else [domains]
        if any(str(domain).lower() in self._config.blocked_domains for domain in domains if domain):
            return None, "blocked_domain"
        return self._home_assistant, ""

    def _handle_user_turn(self, frame: LLMContextFrame) -> bool:
        """Match the finished turn. Return True when the engine takes it over."""
        text = self._user_text()
        if not text:
            return False
        self._start_sync()

        started = time.perf_counter()
        slot_lists = self._home_assistant.slot_lists if self._home_assistant else None
        match = self._matcher.match(text, slot_lists)
        self._metric(METRIC_MATCH_SECS, time.perf_counter() - started)
        if match is None:
            self._record("llm", reason="no_match")
            return False

        target, reason = self._route(match)
        if target is None:
            self._record("llm", intent=match.name, reason=reason)
            return False

        request = IntentRequest(
            name=match.name, data=match.slots, language=self._language, response_key=match.response_key
        )
        if self._config.dry_run:
            would_send = target.build_body(request) if target is self._home_assistant else target.build_call(request)
            logger.info(f"Intent engine dry run: {match.name} -> {target.name} would send {would_send}")
            self._record("llm", intent=match.name, target=target.name, reason="dry_run")
            return False

        self._turn_task = self.create_task(self._run_turn(frame, match, target, request))
        return True

    async def _run_turn(
        self,
        frame: LLMContextFrame,
        match: IntentMatch,
        target: HomeAssistantTarget | ClientToolTarget,
        request: IntentRequest,
    ) -> None:
        """Execute a matched intent, then speak its reply or release the turn to the LLM."""
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(target.handle(request), timeout=target.timeout_secs)
        except (TargetUnavailable, TimeoutError) as exc:
            self._metric(METRIC_TARGET_SECS, time.perf_counter() - started)
            detail = str(exc) or "timeout"
            logger.warning(f"Intent engine: {match.name} via {target.name} failed ({detail}); using the LLM")
            self._record("llm", intent=match.name, target=target.name, reason="target_unavailable")
            if not getattr(exc, "llm_notified", False):
                await self.push_frame(frame)
            return
        except Exception as exc:
            # Never leave a turn unanswered because of an engine bug or a bad template.
            logger.opt(exception=exc).error(f"Intent engine: {match.name} via {target.name} crashed; using the LLM")
            self._record("llm", intent=match.name, target=target.name, reason="engine_error")
            await self.push_frame(frame)
            return
        self._metric(METRIC_TARGET_SECS, time.perf_counter() - started)

        logger.info(f"Intent engine: {match.name} via {target.name} -> {response.response_type}")
        self._record("intent", intent=match.name, target=target.name, response_type=response.response_type)
        if not response.speech:
            return
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(response.speech))
        await self.push_frame(LLMFullResponseEndFrame())

    async def _cancel_turn(self) -> None:
        """Drop the in-flight intent (interruption or shutdown)."""
        if self._client_tools:
            self._client_tools.cancel_pending()
        if self._turn_task and not self._turn_task.done():
            await self.cancel_task(self._turn_task)
        self._turn_task = None

    # -------------------------------------------------------------- telemetry
    def _record(self, handled_by: str, **data: Any) -> None:
        if self._recorder:
            self._recorder.event(EVENT_KIND, processor=_PROCESSOR_NAME, handled_by=handled_by, **data)

    def _metric(self, name: str, value: float) -> None:
        if self._recorder:
            self._recorder.metric(name, value, processor=_PROCESSOR_NAME)


def build_intent_engine(
    config: IntentEngineConfig,
    *,
    language: str,
    context: LLMContext,
    llm: LLMService,
    client_tool_names: tuple[str, ...] = (),
    client_tool_timeout_secs: float = 5.0,
    recorder: TurnTelemetry | None = None,
) -> IntentEngineProcessor | None:
    """Return the engine for a session, or None when it is off or has no sentences."""
    if not config.enabled:
        return None
    use_home_assistant = config.home_assistant_configured
    if not use_home_assistant:
        logger.info("Intent engine: HOME_ASSISTANT_URL or HOME_ASSISTANT_TOKEN is unset; client tools only")

    tool_map = ClientToolMap.load(config.sentences_dir)
    declared = set(client_tool_names)

    def keep_custom_intent(name: str) -> bool:
        # A client-tool intent is active only when the client declared its tool.
        if name in tool_map:
            return tool_map.tool(name) in declared
        return use_home_assistant and name in config.allowed_intents

    matcher = IntentMatcher(
        language,
        allowed_intents=config.allowed_intents,
        sentences_dir=config.sentences_dir,
        wake_words=config.wake_words,
        area=config.area,
        keep_custom_intent=keep_custom_intent,
        load_stock=use_home_assistant,
    )
    if not matcher.has_data:
        logger.info(f"Intent engine: no active sentences for {language}; every turn goes to the LLM")
        return None

    home_assistant = None
    if use_home_assistant:
        home_assistant = HomeAssistantTarget(
            config.home_assistant_url,
            config.home_assistant_token,
            templates=matcher,
            timeout_secs=config.timeout_secs,
            cache_ttl_secs=config.entity_cache_ttl_secs,
            area=config.area,
            blocked_domains=config.blocked_domains,
        )
    client_tools = None
    if any(name in tool_map for name in matcher.custom_intent_names):
        # The registered client-tool handler times out first and tells the LLM; the
        # extra second per call only covers a handler that never reports back.
        client_tools = ClientToolTarget(
            llm, context, tool_map, templates=matcher, timeout_secs=client_tool_timeout_secs + 1.0
        )
    logger.info(
        f"Intent engine enabled: language={language}, home_assistant={use_home_assistant}, "
        f"area={config.area or '(none)'}, dry_run={config.dry_run}, "
        f"timeout={config.timeout_secs:.1f}s, custom_intents={matcher.custom_intent_names or '(none)'}"
    )
    return IntentEngineProcessor(
        config=config,
        language=language,
        matcher=matcher,
        context=context,
        home_assistant=home_assistant,
        client_tools=client_tools,
        tool_map=tool_map,
        recorder=recorder,
    )

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Announce the Matrix messages that arrive during a conversation.

When a contact writes while a session runs, the agent waits until nobody talks
and no question is pending, then says "You just got a new message from Bob.
Shall I read it?" through the intent engine (``MessagingIntents.announcement``):
a yes reads them, any other answer is a turn like any other.

- Only messages that arrive during the session are announced, each one once.
  Messages read meanwhile (by this or another session) are not.
- A message is announced to whoever may read it: a contact with an ``owner`` is
  only announced while that person is known for sure, and waits until then.
- Nothing is said over a pending question (a draft read back, a recording), over
  the user or the bot, or while a reply is on its way. Messages that arrive
  close together are announced together, and an announcement the user talks
  over is said again later: a message counts as announced once it was heard.

The processor sits right before the intent engine; it also feeds the session's
``ConversationActivity`` with the frames it sees.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.shared.conversation_activity import ConversationActivity
from examples.shared.intent_engine.processor import IntentEngineProcessor
from examples.shared.intent_engine.targets import TargetUnavailable
from examples.shared.messaging_intents import MessagingIntents
from examples.shared.messaging_tools import MessagingTools
from messaging.contacts import Contact
from messaging.gateway import MatrixGateway
from messaging.inbox import InboxMessage

POLL_SECS = 0.25
# Messages only the owner may hear wait for them; checked this often meanwhile.
OWNER_WAIT_SECS = 2.0


class MessagingAnnouncer(FrameProcessor):
    """Say that new messages arrived, once the conversation is idle."""

    def __init__(
        self,
        *,
        gateway: MatrixGateway,
        tools: MessagingTools,
        intents: MessagingIntents,
        engine: IntentEngineProcessor,
        activity: ConversationActivity,
    ):
        """Announce ``gateway``'s new messages through ``engine`` with ``intents``' dialog."""
        super().__init__()
        self._gateway = gateway
        self._tools = tools
        self._intents = intents
        self._engine = engine
        self._activity = activity
        self._loop: asyncio.AbstractEventLoop | None = None
        self._remove_listener: Callable[[], None] | None = None
        # Arrived during the session and not announced yet, by event id.
        self._arrivals: dict[str, InboxMessage] = {}
        # Being announced: back to the arrivals when the user talks over it.
        self._announcing: dict[str, InboxMessage] = {}
        self._task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Track the conversation; listen to the gateway while the pipeline runs."""
        await super().process_frame(frame, direction)
        self._activity.on_frame(frame)
        if isinstance(frame, StartFrame):
            self._start()
        elif isinstance(frame, EndFrame | CancelFrame):
            await self.close()
        elif isinstance(frame, InterruptionFrame) and self._announcing:
            for message in self._announcing.values():
                self._arrived(message)
            self._announcing.clear()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._announcing.clear()
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        """Stop listening."""
        await super().cleanup()
        await self.close()

    def _start(self) -> None:
        if self._remove_listener is None:
            self._loop = asyncio.get_running_loop()
            self._remove_listener = self._gateway.add_listener(self._on_message)

    async def close(self) -> None:
        """Stop listening and drop what was not announced (idempotent)."""
        if self._remove_listener is not None:
            self._remove_listener()
            self._remove_listener = None
        self._arrivals.clear()
        self._announcing.clear()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    # --------------------------------------------------------------- arrivals
    def _on_message(self, message: InboxMessage, contact: Contact) -> None:
        """Gateway listener: runs in the sync loop, so it only schedules the announcement."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._arrived, message)

    def _arrived(self, message: InboxMessage) -> None:
        if self._remove_listener is None:
            return
        self._arrivals[message.event_id] = message
        if self._task is None or self._task.done():
            self._task = self.create_task(self._announce_when_idle(), name="matrix-announce")

    def _can_speak(self) -> bool:
        return self._activity.idle() and not self._engine.busy

    async def _announce_when_idle(self) -> None:
        while self._arrivals:
            if not self._can_speak():
                await asyncio.sleep(POLL_SECS)
                continue
            if not await self._announce():
                await asyncio.sleep(OWNER_WAIT_SECS)

    async def _announce(self) -> bool:
        """Announce the due arrivals now; False when they all wait for their owner."""
        unread = await self._tools.still_unread(list(self._arrivals))
        for event_id in [e for e in self._arrivals if e not in unread]:
            del self._arrivals[event_id]
        allowed = {contact.matrix_id: contact for contact in self._tools.allowed_contacts()}
        due = [m for m in sorted(self._arrivals.values(), key=lambda m: m.sent_at) if m.sender in allowed]
        if not due:
            return not self._arrivals
        senders = list(dict.fromkeys(m.sender for m in due))
        by_sender = {contact.matrix_id: n for contact, n in await self._tools.unread()}
        unread_counts = [(allowed[s], by_sender.get(s, 0)) for s in senders]
        unread_counts = [(contact, n) for contact, n in unread_counts if n]
        if not unread_counts:
            return True
        # Counting awaited: somebody may have started talking meanwhile.
        if not self._can_speak():
            return True
        try:
            response = self._intents.announcement(unread_counts, len(due))
        except TargetUnavailable as exc:
            logger.warning(f"Matrix: cannot announce new messages ({exc})")
            self._arrivals.clear()
            return True
        if not self._engine.speak_first(response):
            return True
        self._activity.expect_reply()
        for message in due:
            self._announcing[message.event_id] = self._arrivals.pop(message.event_id)
        logger.info(f"Matrix: announced new messages from {', '.join(c.name for c, _ in unread_counts)}")
        return True

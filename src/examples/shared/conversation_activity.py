# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who is talking in a session, so the agent only speaks first while nobody does.

One ``ConversationActivity`` is shared by everything that may speak without being
asked (the face-ID greeting, the Matrix message announcements): each one checks
``idle()`` and calls ``expect_reply()`` in the same synchronous step before it
speaks, so two of them never talk over each other in the same quiet moment.
"""

from __future__ import annotations

import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMContextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

# Speaking first needs this much silence after the last user or bot activity.
QUIET_SECS = 1.5
# After a reply was started (an LLM run, a spoken announcement), the bot is expected to speak meanwhile.
RESPONSE_PENDING_SECS = 15.0


class ConversationActivity:
    """Conversation activity of one session, fed with the frames a processor sees."""

    def __init__(self, *, quiet_secs: float = QUIET_SECS):
        """Start idle-tracking now (the session just started: not quiet yet)."""
        self.quiet_secs = quiet_secs
        self.user_speaking = False
        self.bot_speaking = False
        self._response_pending_until = 0.0
        self._last_activity_at = time.monotonic()

    def on_frame(self, frame: Frame) -> None:
        """Track a frame: user and bot speech, and the user turns that reach the LLM."""
        now = time.monotonic()
        if isinstance(frame, UserStartedSpeakingFrame):
            self.user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.user_speaking = False
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_speaking = False
            self._response_pending_until = 0.0
        elif isinstance(frame, LLMContextFrame):
            self._response_pending_until = now + RESPONSE_PENDING_SECS
        else:
            return
        self._last_activity_at = now

    def touch(self) -> None:
        """Something happened in the conversation (a user turn started)."""
        self._last_activity_at = time.monotonic()

    def expect_reply(self) -> None:
        """A reply was just started: nobody else speaks first until the bot has spoken."""
        self._response_pending_until = time.monotonic() + RESPONSE_PENDING_SECS

    def idle(self, now: float | None = None) -> bool:
        """Whether nobody is talking and no reply is expected, for long enough to speak first."""
        now = time.monotonic() if now is None else now
        if self.user_speaking or self.bot_speaking or now < self._response_pending_until:
            return False
        return now - self._last_activity_at >= self.quiet_secs

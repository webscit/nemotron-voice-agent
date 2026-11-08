# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the microphone audio of the last user turn, for voice messages.

Place it after the user aggregator and right before the intent engine. It
records the input audio from ``UserStartedSpeakingFrame`` (plus a short
pre-roll, since speech starts before it is detected) until the turn ends: at
``UserStoppedSpeakingFrame``, or when the turn's ``LLMContextFrame`` passes,
whichever comes first. Because the context frame passes here before it
reaches the intent engine, the engine always finds the audio of the turn it is
handling.
"""

from __future__ import annotations

from collections import deque

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.shared.intent_engine.targets import AudioClip

PREROLL_SECS = 0.5
# A turn longer than this keeps only its beginning.
MAX_TURN_SECS = 120.0


class UserTurnAudioCapture(FrameProcessor):
    """Record each user turn's audio; ``take_last_turn`` hands out the last one."""

    def __init__(self, *, preroll_secs: float = PREROLL_SECS, max_turn_secs: float = MAX_TURN_SECS):
        """Set the pre-roll and the longest recording."""
        super().__init__()
        self._preroll_secs = preroll_secs
        self._max_turn_secs = max_turn_secs
        self._sample_rate = 0
        self._preroll: deque[bytes] = deque()
        self._preroll_bytes = 0
        self._turn: bytearray | None = None
        self._last: AudioClip | None = None

    def take_last_turn(self) -> AudioClip | None:
        """The audio of the last finished user turn, once (None when there is none)."""
        clip, self._last = self._last, None
        return clip

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Record input audio around user turns; every frame is forwarded."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._sample_rate = frame.audio_in_sample_rate
        elif isinstance(frame, InputAudioRawFrame):
            self._on_audio(frame)
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._turn = bytearray(b"".join(self._preroll))
            self._last = None
        elif isinstance(frame, UserStoppedSpeakingFrame) or (
            isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM
        ):
            self._finish_turn()
        await self.push_frame(frame, direction)

    def _on_audio(self, frame: InputAudioRawFrame) -> None:
        self._sample_rate = frame.sample_rate
        max_bytes = int(self._max_turn_secs * frame.sample_rate) * 2 * frame.num_channels
        if self._turn is not None and len(self._turn) < max_bytes:
            self._turn += frame.audio[: max_bytes - len(self._turn)]
        self._preroll.append(frame.audio)
        self._preroll_bytes += len(frame.audio)
        preroll_max = int(self._preroll_secs * frame.sample_rate) * 2 * frame.num_channels
        while self._preroll and self._preroll_bytes - len(self._preroll[0]) >= preroll_max:
            self._preroll_bytes -= len(self._preroll.popleft())

    def _finish_turn(self) -> None:
        if self._turn is None:
            return
        if self._turn and self._sample_rate:
            self._last = AudioClip(bytes(self._turn), self._sample_rate)
        self._turn = None

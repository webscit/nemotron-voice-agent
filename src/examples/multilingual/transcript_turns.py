# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turns opened by a final transcript that VAD missed.

A one-syllable answer ("oui", "non") is often shorter than the VAD ``start_secs``
(0.2 s): VAD never reports the user speaking, but the ASR still transcribes it.
With VAD-only turn starts, that transcript waited in the user aggregator and was
glued to the front of the next turn ("Oui. Lis mes nouveaux messages").
"""

import math
import time

from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, TranscriptionFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.transcription_user_turn_start_strategy import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy

# How long a transcript-only turn waits for VAD to report more speech before it ends.
TRANSCRIPT_TURN_WAIT_SECS = 0.3
# Transcripts are ignored while the bot speaks and for this long after: the browser
# plays audio slightly behind the server, and echo of the bot must not start a turn.
BOT_SPEECH_TAIL_SECS = 0.3


class FinalTranscriptUserTurnStartStrategy(TranscriptionUserTurnStartStrategy):
    """Start a user turn from a final transcript, but never while the bot is speaking.

    Placed after ``VADUserTurnStartStrategy``: when VAD already opened the turn,
    the controller ignores the second start. Interim transcripts never start a turn.
    """

    def __init__(self, *, bot_speech_tail_secs: float = BOT_SPEECH_TAIL_SECS, **kwargs):
        """Ignore transcripts until ``bot_speech_tail_secs`` after the bot stops speaking."""
        super().__init__(use_interim=False, **kwargs)
        self._bot_speech_tail_secs = bot_speech_tail_secs
        self._bot_speaking = False
        self._bot_stopped_at = -math.inf

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Track the bot's speech, and start a turn on a final transcript outside it."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = time.monotonic()
        elif isinstance(frame, TranscriptionFrame):
            if self._bot_speaking or time.monotonic() - self._bot_stopped_at < self._bot_speech_tail_secs:
                return ProcessFrameResult.CONTINUE
            logger.debug(f"{self}: final transcript without a turn; starting one: [{frame.text}]")
        return await super().process_frame(frame)


class TranscriptTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """Smart Turn stop strategy that ends a transcript-only turn quickly.

    Pipecat already ends a turn whose transcript arrived without VAD activity, but
    only after ``ttfs_p99_latency - stop_secs`` (0.8 s with the NVIDIA defaults).
    The transcript is final by then, so the wait only lets VAD report that the user
    is still speaking: ``wait_secs`` is enough. Turns VAD saw are unchanged.

    Relies on private state of ``TurnAnalyzerUserTurnStopStrategy`` (Pipecat 1.7):
    ``tests/unit/test_transcript_turns.py`` exercises it through the real aggregator.
    """

    def __init__(self, *, wait_secs: float = TRANSCRIPT_TURN_WAIT_SECS, **kwargs):
        """End a transcript-only turn ``wait_secs`` after its final transcript."""
        super().__init__(**kwargs)
        self._transcript_turn_wait_secs = wait_secs

    async def _handle_transcription(self, frame: TranscriptionFrame):
        await super()._handle_transcription(frame)
        transcript_only = not self._vad_user_speaking and not self._vad_stopped
        if frame.finalized and transcript_only and self._timeout_task:
            await self.task_manager.cancel_task(self._timeout_task)
            self._timeout_task = self.task_manager.create_task(
                self._timeout_handler(self._transcript_turn_wait_secs), f"{self}::_timeout_handler"
            )

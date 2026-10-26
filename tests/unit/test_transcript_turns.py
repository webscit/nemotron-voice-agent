# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Turns opened by a final transcript that VAD missed, through the real user aggregator."""

import asyncio
import time

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InterimTranscriptionFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from examples.multilingual.transcript_turns import (
    BOT_SPEECH_TAIL_SECS,
    TRANSCRIPT_TURN_WAIT_SECS,
    FinalTranscriptUserTurnStartStrategy,
    TranscriptTurnStopStrategy,
)

# What the NVIDIA ASR advertises (Pipecat's NVIDIA_TTFS_P99): without the shorter
# wait, a transcript-only turn would end 1.0 - stop_secs after its transcript.
NVIDIA_TTFS_P99 = 1.0


class FakeTurnAnalyzer(BaseTurnAnalyzer):
    """Smart Turn stand-in that never decides on its own: turns end through the transcript path."""

    @property
    def speech_triggered(self) -> bool:
        return False

    @property
    def params(self) -> BaseTurnParams:
        return BaseTurnParams()

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self):
        return EndOfTurnState.INCOMPLETE, None

    def clear(self):
        pass


class Harness:
    def __init__(self):
        self.started: list[tuple[float, str]] = []
        self.stopped: list[tuple[float, str]] = []
        self.aggregator = LLMUserAggregator(
            LLMContext(),
            params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    start=[VADUserTurnStartStrategy(), FinalTranscriptUserTurnStartStrategy()],
                    stop=[TranscriptTurnStopStrategy(turn_analyzer=FakeTurnAnalyzer())],
                ),
            ),
        )

        @self.aggregator.event_handler("on_user_turn_started")
        async def _started(aggregator, strategy):
            self.started.append((time.monotonic(), type(strategy).__name__))

        @self.aggregator.event_handler("on_user_turn_stopped")
        async def _stopped(aggregator, strategy, message):
            self.stopped.append((time.monotonic(), message.content))

        self.worker = PipelineWorker(Pipeline([self.aggregator]), cancel_on_idle_timeout=False, enable_rtvi=False)

    async def __aenter__(self):
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self._task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.05)
        await self.send(STTMetadataFrame(service_name="stt", ttfs_p99_latency=NVIDIA_TTFS_P99))
        return self

    async def __aexit__(self, *exc):
        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self._task, timeout=5)

    async def send(self, frame):
        await self.worker.queue_frame(frame)
        await asyncio.sleep(0.02)

    async def say(self, text: str) -> float:
        """A final transcript, as the NVIDIA ASR sends it; returns when it was sent."""
        sent = time.monotonic()
        await self.send(TranscriptionFrame(text, "user", "now", finalized=True))
        return sent


def run(coro):
    asyncio.run(coro)


def test_final_transcript_vad_missed_is_its_own_turn():
    async def scenario():
        async with Harness() as h:
            sent = await h.say(" Oui.")
            await asyncio.sleep(TRANSCRIPT_TURN_WAIT_SECS + 0.3)
            assert [name for _, name in h.started] == ["FinalTranscriptUserTurnStartStrategy"]
            assert [text for _, text in h.stopped] == ["Oui."]
            # The shorter wait, not Pipecat's 0.8 s.
            assert h.stopped[0][0] - sent < TRANSCRIPT_TURN_WAIT_SECS + 0.2
            # The next utterance is a turn of its own: nothing is glued to it.
            await h.say(" Lis mes nouveaux messages.")
            await asyncio.sleep(TRANSCRIPT_TURN_WAIT_SECS + 0.3)
            assert [text for _, text in h.stopped] == ["Oui.", "Lis mes nouveaux messages."]

    run(scenario())


def test_interim_transcript_does_not_start_a_turn():
    async def scenario():
        async with Harness() as h:
            await h.send(InterimTranscriptionFrame(" Ou", "user", "now"))
            await asyncio.sleep(0.3)
            assert h.started == []

    run(scenario())


def test_transcript_while_the_bot_speaks_or_just_after_does_not_start_a_turn():
    async def scenario():
        async with Harness() as h:
            await h.send(BotStartedSpeakingFrame())
            await h.say(" Alors")  # echo of the bot, or a word over it: VAD handles barge-in
            await h.send(BotStoppedSpeakingFrame())
            await h.say(" Alors")  # still inside the playback tail
            await asyncio.sleep(0.1)
            assert h.started == []
            await asyncio.sleep(BOT_SPEECH_TAIL_SECS)
            await h.say(" Oui")
            await asyncio.sleep(TRANSCRIPT_TURN_WAIT_SECS + 0.3)
            assert [name for _, name in h.started] == ["FinalTranscriptUserTurnStartStrategy"]
            assert len(h.stopped) == 1
            assert h.stopped[0][1].endswith("Oui")

    run(scenario())


def test_vad_reporting_speech_keeps_the_turn_open():
    async def scenario():
        async with Harness() as h:
            await h.say(" Oui")
            # VAD catches up: the user is still speaking, so the quick end is cancelled.
            await h.send(VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=time.time()))
            await asyncio.sleep(TRANSCRIPT_TURN_WAIT_SECS + 0.5)
            assert len(h.started) == 1
            assert h.stopped == []

    run(scenario())

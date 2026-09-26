# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import asyncio
import wave

from pipecat.frames.frames import (
    AudioBufferStartRecordingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.tests.utils import SleepFrame, run_test

from monitoring.config import MonitoringConfig
from monitoring.recorder import SessionRecorder
from monitoring.store import LocalArtifactStore, SessionStore


def test_recorder_inside_a_real_pipeline(tmp_path):
    """Real worker + observer dispatch + AudioBufferProcessor → rows and WAVs."""
    settings = MonitoringConfig(
        enabled=True,
        data_dir=tmp_path,
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        record_audio_turns=True,
        record_audio_stereo=True,
        record_video="off",
        video_fps=1.0,
    )
    store = SessionStore(settings.db_url)
    store.create_schema()

    async def scenario():
        recorder = SessionRecorder(
            session_id="live1",
            example="multilingual-assistant",
            config={},
            settings=settings,
            store=store,
            artifacts=LocalArtifactStore(settings.artifacts_dir),
        )
        await recorder.start()
        (audio_processor,) = recorder.processors()
        mic = [InputAudioRawFrame(audio=bytes(320 * 2), sample_rate=16000, num_channels=1) for _ in range(25)]
        tts = [OutputAudioRawFrame(audio=bytes(441 * 2), sample_rate=22050, num_channels=1) for _ in range(20)]
        await run_test(
            audio_processor,
            frames_to_send=[
                AudioBufferStartRecordingFrame(),
                SleepFrame(0.05),  # control frame: let it land before the system frames
                UserStartedSpeakingFrame(),
                *mic,
                UserStoppedSpeakingFrame(),
                SleepFrame(0.05),
                BotStartedSpeakingFrame(),
                *tts,
                SleepFrame(0.05),  # the transport only reports bot-stopped after playback
                BotStoppedSpeakingFrame(),
                MetricsFrame(data=[TTFBMetricsData(processor="NvidiaTTSService#0", value=0.21)]),
                SleepFrame(0.1),
            ],
            observers=recorder.observers(),
        )
        await asyncio.sleep(0.1)  # let the event-handler tasks enqueue
        await recorder.close()

    asyncio.run(scenario())

    media = store.rows("media", "live1")
    by_modality = {m["modality"]: m for m in media}
    assert {"audio_user", "audio_bot", "audio_conversation"} <= set(by_modality)
    assert by_modality["audio_user"]["sample_rate"] == 16000
    with wave.open(str(settings.artifacts_dir / by_modality["audio_conversation"]["artifact_key"])) as wf:
        assert wf.getnchannels() == 2 and wf.getframerate() == 16000 and wf.getnframes() > 0
    # Metrics frames are reported at every hop; the observer stores each once.
    ttfb = [m for m in store.rows("metrics", "live1") if m["name"] == "ttfb"]
    assert len(ttfb) == 1 and ttfb[0]["value"] == 0.21
    kinds = [e["kind"] for e in store.rows("events", "live1")]
    assert kinds.count("user_started_speaking") == 1 and kinds.count("bot_stopped_speaking") == 1
    assert store.get_session("live1")["end_reason"] == "completed"

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import base64
import io
import wave
from dataclasses import replace

from PIL import Image
from pipecat.frames.frames import (
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    UserImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from monitoring.config import MonitoringConfig
from monitoring.media import extract_inline_media
from monitoring.recorder import SessionRecorder, flatten_metrics
from monitoring.store import LocalArtifactStore, SessionStore


class FakeProcessor:
    def __init__(self, name: str):
        self.name = name


class FakeEmitter:
    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def decorator(fn):
            self.handlers[name] = fn
            return fn

        return decorator

    async def emit(self, name, *args):
        await self.handlers[name](self, *args)


def _png_data_url(color=(255, 0, 0)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _settings(tmp_path) -> MonitoringConfig:
    return MonitoringConfig(
        enabled=True,
        data_dir=tmp_path,
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        record_audio_turns=True,
        record_audio_stereo=True,
        record_video="off",
        video_fps=1.0,
    )


def _push(observer, frame, source, destination):
    return observer.on_push_frame(
        FramePushed(
            source=source, destination=destination, frame=frame, direction=FrameDirection.DOWNSTREAM, timestamp=0
        )
    )


def test_flatten_metrics_names():
    assert flatten_metrics(TTFBMetricsData(processor="llm", value=0.25)) == [("ttfb", 0.25)]
    usage = LLMUsageMetricsData(
        processor="llm", value=LLMTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    names = dict(flatten_metrics(usage))
    assert names["llm_usage.prompt_tokens"] == 10
    assert names["llm_usage.completion_tokens"] == 5


def test_extract_inline_media_replaces_data_urls():
    url = _png_data_url()
    stripped, found = extract_inline_media(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}, {"result": url}]
    )
    assert len(found) == 2 and found[0].sha256 == found[1].sha256
    assert stripped[0]["content"][0]["image_url"]["url"]["type"] == "media_ref"
    assert stripped[1]["result"]["mime"] == "image/png"


def test_recorder_end_to_end(tmp_path):
    settings = _settings(tmp_path)
    store = SessionStore(settings.db_url)
    store.create_schema()
    artifacts = LocalArtifactStore(settings.artifacts_dir)
    llm = FakeProcessor("NvidiaLLMService#0")
    upstream, tts = FakeProcessor("reminder"), FakeProcessor("tts")

    async def scenario():
        recorder = SessionRecorder(
            session_id="s1",
            example="multilingual-assistant",
            config={"llm": {"model": "nemotron"}},
            settings=settings,
            store=store,
            artifacts=artifacts,
            llm=llm,
            language="de-DE",
        )
        user_agg, assistant_agg, latency = FakeEmitter(), FakeEmitter(), FakeEmitter()
        recorder.attach(object(), user_agg, assistant_agg, latency)
        await recorder.start()
        observer = recorder.observers()[0]

        await user_agg.emit("on_user_turn_started", None)
        await user_agg.emit("on_user_turn_stopped", None, type("M", (), {"content": "Was siehst du?"})())

        context = LLMContext([{"role": "system", "content": "sys"}])
        context.add_message({"role": "user", "content": [{"type": "image_url", "image_url": {"url": _png_data_url()}}]})
        await _push(observer, LLMContextFrame(context=context), upstream, llm)
        await _push(observer, MetricsFrame(data=[TTFBMetricsData(processor=llm.name, value=0.4)]), llm, tts)
        await _push(observer, LLMTextFrame(text="Ein rotes "), llm, tts)
        await _push(observer, LLMTextFrame(text="Bild."), llm, tts)
        await _push(observer, LLMFullResponseEndFrame(), llm, tts)
        await _push(
            observer,
            FunctionCallResultFrame(
                function_name="take_photo",
                tool_call_id="c1",
                arguments={},
                result={"image": _png_data_url((0, 0, 255))},
            ),
            llm,
            tts,
        )
        await _push(observer, UserImageRawFrame(image=bytes(4 * 4 * 3), size=(4, 4), format="RGB"), upstream, llm)
        await latency.emit("on_latency_measured", 0.9)
        await assistant_agg.emit("on_assistant_turn_started")
        await assistant_agg.emit(
            "on_assistant_turn_stopped", type("M", (), {"content": "Ein rotes Bild.", "interrupted": False})()
        )
        recorder._queue_turn_audio("user", 1, b"\x00\x01" * 1600, 16000)
        stereo = recorder._stereo
        recorder.writer.put_task(lambda a: stereo.append(b"\x00\x00" * 3200, 16000, 2) or [])
        await recorder.close()

    asyncio.run(scenario())

    session = store.get_session("s1")
    assert session["ended_at"] is not None and session["config"]["llm"]["model"] == "nemotron"
    turns = store.rows("turns", "s1")
    assert [t["idx"] for t in turns] == [0, 1]
    assert turns[1]["user_text"] == "Was siehst du?" and turns[1]["bot_text"] == "Ein rotes Bild."

    (call,) = store.rows("llm_calls", "s1")
    assert call["output_text"] == "Ein rotes Bild." and call["ttfb"] == 0.4
    assert call["n_images"] == 1 and call["image_pixels"] == 48
    assert call["messages"][1]["content"][0]["image_url"]["url"]["type"] == "media_ref"

    media = {(m["modality"], m["source"]) for m in store.rows("media", "s1")}
    assert ("image", "context") in media
    assert ("image", "tool:take_photo") in media
    assert ("image", "user_image") in media
    assert ("audio_user", "pipeline") in media
    assert ("audio_conversation", "pipeline") in media
    for row in store.rows("media", "s1"):
        assert (settings.artifacts_dir / row["artifact_key"]).exists()

    conversation = next(m for m in store.rows("media", "s1") if m["modality"] == "audio_conversation")
    with wave.open(str(settings.artifacts_dir / conversation["artifact_key"])) as wf:
        assert wf.getnchannels() == 2 and wf.getnframes() == 1600

    metric_names = {m["name"] for m in store.rows("metrics", "s1")}
    assert {"ttfb", "user_bot_latency"} <= metric_names
    result_event = next(e for e in store.rows("events", "s1") if e["kind"] == "function_result")
    assert result_event["data"]["result"]["image"]["type"] == "media_ref"


def test_disabled_recorder_returns_none(tmp_path):
    settings = replace(_settings(tmp_path), enabled=False)
    assert SessionRecorder.create(session_id="x", example="e", config={}, settings=settings) is None

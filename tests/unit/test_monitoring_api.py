# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from monitoring.api import create_review_router
from monitoring.config import MonitoringConfig
from monitoring.jobs.runner import _DEFAULTS
from monitoring.media import pcm16_to_wav
from monitoring.store import LocalArtifactStore, SessionStore

DREAMER = {**_DEFAULTS, "reasr": {"reference": {"name": "voxtral", "protocol": "openai"}, "candidates": []}}


@pytest.fixture
def env(tmp_path):
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
    artifacts = LocalArtifactStore(settings.artifacts_dir)
    now = time.time()
    store.create_session(
        {
            "id": "s1",
            "example": "multilingual-assistant",
            "started_at": now - 100,
            "last_seen_at": now - 50,
            "ended_at": now - 50,
            "config": {"language": "fr-FR", "asr": {"model": "nemo"}},
            "artifact_prefix": "sessions/s1/",
        }
    )
    turns = {1: ("allume la lumière", "allume la lumière"), 2: ("il fait beau", "quel temps fait-il")}
    rows = []
    for idx, (live, _voxtral) in turns.items():
        key = f"sessions/s1/turns/{idx:03d}_user_00.wav"
        artifacts.put(key, pcm16_to_wav(bytes(range(256)) * 125, 16000))
        rows += [
            ("turns", {"session_id": "s1", "idx": idx, "user_text": live, "user_started_at": now - 90 + idx}),
            (
                "media",
                {
                    "session_id": "s1",
                    "turn_idx": idx,
                    "ts": now,
                    "modality": "audio_user",
                    "sha256": key,
                    "artifact_key": key,
                    "duration_secs": 1.0,
                },
            ),
        ]
    store.write_batch(rows)
    store.add_annotations(
        [
            {
                "session_id": "s1",
                "target_type": "turn",
                "target_id": f"s1:{idx}",
                "source": "voxtral",
                "kind": "transcript",
                "value": {"text": voxtral},
            }
            for idx, (_, voxtral) in turns.items()
        ]
    )
    app = FastAPI()
    app.include_router(create_review_router(settings, store, artifacts, DREAMER))
    return TestClient(app), store


def test_sessions_and_detail(env):
    client, _ = env
    listing = client.get("/api/review/sessions").json()
    assert listing["total"] == 1
    (summary,) = listing["sessions"]
    assert summary["user_turns"] == 2 and summary["audio_turns"] == 2 and summary["reviewed_turns"] == 0
    assert summary["language"] == "fr-FR"

    detail = client.get("/api/review/sessions/s1").json()
    turn = next(t for t in detail["turns"] if t["idx"] == 2)
    assert turn["transcripts"] == {"live:nemo": "il fait beau", "voxtral": "quel temps fait-il"}
    assert turn["audio"]["user"][0]["key"].endswith("002_user_00.wav")
    assert client.get("/api/review/sessions/nope").status_code == 404


def test_artifact_supports_range(env):
    client, _ = env
    url = "/api/review/artifacts/sessions/s1/turns/001_user_00.wav"
    full = client.get(url)
    assert full.status_code == 200 and full.content[:4] == b"RIFF"
    partial = client.get(url, headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206 and len(partial.content) == 100
    assert client.get("/api/review/artifacts/../../etc/passwd").status_code == 404


def test_asr_queue_orders_by_disagreement_and_reference_rescoring(env):
    client, store = env
    queue = client.get("/api/review/asr/queue").json()
    assert queue["open"] == 2 and queue["agreeing"] == 1
    assert [i["turn_idx"] for i in queue["items"]] == [2, 1]  # disagreement first
    assert queue["items"][0]["disagreement"] > 0 and queue["items"][1]["agree"]

    saved = client.post(
        "/api/review/asr/references",
        json={"annotator": "fred", "items": [{"session_id": "s1", "turn_idx": 2, "text": "il fait beau"}]},
    ).json()
    assert saved["saved"] == 1
    live = saved["wer_summary"]["s1"]["live:nemo"]
    # turn 2 now uses the human reference (live correct), turn 1 still the voxtral one (agree).
    assert live["wer"] == 0.0 and live["turns"] == 2
    assert sorted(live["reference_sources"]) == ["human:fred", "voxtral"]

    queue = client.get("/api/review/asr/queue").json()
    assert queue["open"] == 1 and queue["total"] == 2 and [i["turn_idx"] for i in queue["items"]] == [1]
    reviewed = client.get("/api/review/asr/queue", params={"status": "reviewed"}).json()
    assert reviewed["items"][0]["human_reference"]["text"] == "il fait beau"
    # Rescoring replaces derived rows rather than appending.
    client.post(
        "/api/review/asr/references",
        json={"annotator": "fred", "items": [{"session_id": "s1", "turn_idx": 1, "text": "allume la lumière"}]},
    )
    assert len(store.annotations_for("s1", kind="wer_summary")) == 1  # one live summary, not one per save
    assert client.get("/api/review/activities").json()["activities"][0]["open"] == 0


def test_validation_and_dreamer(env):
    client, store = env
    bad = client.post(
        "/api/review/asr/references",
        json={"annotator": "<script>", "items": [{"session_id": "s1", "turn_idx": 1, "text": "x"}]},
    )
    assert bad.status_code == 422
    assert client.post("/api/review/jobs", json={"kind": "nope", "session_ids": ["s1"]}).status_code == 422
    assert client.post("/api/review/jobs", json={"kind": "reasr", "session_ids": ["s1"]}).json() == {"queued": 1}
    status = client.get("/api/review/dreamer").json()
    assert status["live_sessions"] == 0 and status["pending"] == 1

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import pytest

from monitoring.jobs import JOB_REGISTRY, Job, register
from monitoring.jobs import reasr as reasr_module
from monitoring.jobs.runner import _DEFAULTS, Dreamer
from monitoring.jobs.wer import error_rates, normalize
from monitoring.media import pcm16_to_wav
from monitoring.store import LocalArtifactStore, SessionStore


class FakeClock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeServices:
    def __init__(self):
        self.running: list[str] = []
        self.started: list[str] = []

    def ensure_running(self, service, **_):
        if service not in self.running:
            self.running.append(service)
            self.started.append(service)

    def stop_started(self):
        self.running.clear()


@pytest.fixture
def stores(tmp_path):
    store = SessionStore(f"sqlite:///{tmp_path / 'db.sqlite'}")
    store.create_schema()
    return store, LocalArtifactStore(tmp_path / "artifacts")


def _session(store, sid, *, started, ended=None, language="en-US"):
    store.create_session(
        {
            "id": sid,
            "example": "multilingual-assistant",
            "started_at": started,
            "last_seen_at": ended or started,
            "ended_at": ended,
            "config": {"language": language, "asr": {"model": "live-asr"}},
            "artifact_prefix": f"sessions/{sid}/",
        }
    )


@register
class StepJob(Job):
    kind = "test_steps"
    steps = 3
    on_step = None

    def required_services(self, ctx):
        return ["alt-model"]

    def run(self, ctx):
        for step in range(ctx.progress.get("next", 0), self.steps):
            ctx.check_preempted()
            if StepJob.on_step:
                StepJob.on_step(step)
            ctx.save_progress(next=step + 1)


def _dreamer(stores, clock, **overrides):
    store, artifacts = stores
    config = {**_DEFAULTS, "auto_enqueue": ["test_steps"], "idle_grace_secs": 60, **overrides}
    return Dreamer(store, artifacts, config, services=FakeServices(), clock=clock)


def test_wer_normalization_and_counts():
    assert normalize("Hallo, Welt!") == "hallo welt"
    rates = error_rates("the cat sat on the mat", "the cat sat on mat")
    assert rates["word_errors"] == 1 and rates["ref_words"] == 6
    assert error_rates("", "")["wer"] == 0.0


def test_dreamer_waits_for_idle_grace_and_live_sessions(stores):
    store, _ = stores
    clock = FakeClock()
    _session(store, "done", started=clock.now - 100, ended=clock.now - 10)
    _session(store, "live", started=clock.now - 5)
    dreamer = _dreamer(stores, clock)

    assert not dreamer.run_once()  # a live session blocks everything
    assert store.jobs()[0]["status"] == "pending"

    store.end_session("live", clock.now, "completed")
    clock.now += 30
    assert not dreamer.run_once()  # still inside the idle grace period
    clock.now += 31
    assert dreamer.run_once()
    assert dreamer.run_once()
    assert {j["target"]: j["status"] for j in store.jobs()} == {"done": "done", "live": "done"}
    assert dreamer.services.running == ["alt-model"]  # kept up while jobs remain
    assert not dreamer.run_once()
    assert dreamer.services.running == []  # stopped once the queue drained


def test_dreamer_preempts_and_resumes(stores):
    store, _ = stores
    clock = FakeClock()
    _session(store, "s1", started=clock.now - 500, ended=clock.now - 400)
    dreamer = _dreamer(stores, clock)
    seen = []

    def on_step(step):
        seen.append(step)
        if step == 1 and len(seen) == 2:
            _session(store, "caller", started=clock.now)  # a user connects mid-job
            clock.now += 2  # let the rate-limited preemption check fire

    StepJob.on_step = on_step
    try:
        assert dreamer.run_once()
        (job,) = [j for j in store.jobs() if j["target"] == "s1"]
        assert job["status"] == "pending" and job["attempts"] == 0 and job["progress"] == {"next": 2}
        assert dreamer.services.running == []

        store.end_session("caller", clock.now, "completed")
        clock.now += 120
        while dreamer.run_once():
            pass
    finally:
        StepJob.on_step = None
    # s1 resumed from its checkpoint (step 2), then the caller's own session ran.
    assert seen == [0, 1, 2, 0, 1, 2]
    assert {j["target"]: j["status"] for j in store.jobs()} == {"s1": "done", "caller": "done"}


def test_orphaned_session_is_closed(stores):
    store, _ = stores
    clock = FakeClock()
    _session(store, "crashed", started=clock.now - 1000)
    dreamer = _dreamer(stores, clock)
    dreamer.startup()
    assert store.get_session("crashed")["end_reason"] == "orphaned"


def test_reasr_scores_live_and_candidates(stores, monkeypatch):
    store, artifacts = stores
    clock = FakeClock()
    _session(store, "s1", started=clock.now - 500, ended=clock.now - 400, language="en-US")
    _session(store, "s2", started=clock.now - 300, ended=clock.now - 200, language="de-DE")
    for sid in ("s1", "s2"):
        key = f"sessions/{sid}/turns/001_user_00.wav"
        artifacts.put(key, pcm16_to_wav(b"\x00\x00" * 1600, 16000))
        store.write_batch(
            [
                (
                    "media",
                    {
                        "session_id": sid,
                        "turn_idx": 1,
                        "ts": clock.now,
                        "modality": "audio_user",
                        "sha256": sid,
                        "artifact_key": key,
                    },
                ),
                ("turns", {"session_id": sid, "idx": 1, "user_text": "turn the light on"}),
            ]
        )
    outputs = {"reference": "turn the light on please", "english-alt": "turn the lights on please"}

    class FakeTranscriber:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def wait_ready(self):
            pass

        def transcribe_wav(self, wav, language):
            return outputs[self.endpoint.name]

    monkeypatch.setattr(reasr_module, "RivaTranscriber", FakeTranscriber)
    config = {
        "auto_enqueue": ["reasr"],
        "reasr": {
            "reference": {"name": "reference", "server": "ref:1"},
            "candidates": [{"name": "english-alt", "server": "alt:1", "service": "alt-svc", "languages": ["en"]}],
        },
    }
    dreamer = _dreamer(stores, clock, **config)
    clock.now += 1000
    while dreamer.run_once():
        pass
    assert dreamer.services.started == ["alt-svc"]  # not started for the German session

    summaries = {a["source"]: a["value"] for a in store.annotations_for("s1", kind="wer_summary")}
    assert summaries["live:live-asr"]["wer"] == pytest.approx(1 / 5)
    assert summaries["english-alt"]["wer"] == pytest.approx(1 / 5)
    assert {a["source"] for a in store.annotations_for("s2", kind="wer_summary")} == {"live:live-asr"}

    # A human correction becomes the reference on the next run.
    store.add_annotations(
        [
            {
                "session_id": "s1",
                "target_type": "turn",
                "target_id": "s1:1",
                "source": "human:qa",
                "kind": "transcript",
                "value": {"text": "turn the light on"},
            }
        ]
    )
    store.reset_job("reasr", "s1")
    while dreamer.run_once():
        pass
    latest = {a["source"]: a["value"] for a in store.annotations_for("s1", kind="wer_summary")}
    assert latest["live:live-asr"]["wer"] == 0.0
    assert latest["live:live-asr"]["reference_sources"] == ["human:qa"]


def test_registry_contains_reasr():
    assert "reasr" in JOB_REGISTRY

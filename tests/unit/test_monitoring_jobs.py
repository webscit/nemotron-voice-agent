# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import json

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

    monkeypatch.setattr(reasr_module, "make_transcriber", FakeTranscriber)
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


EMOTION_QUESTION = {
    "type": "choice",
    "instructions": "Which emotion?",
    "criteria": {"happy": "Joy", "sad": "Sadness", "none": "No emotion"},
}


def _decision_session(store, sid, *, started, ended, turns):
    """A session that ran ``play_emotion`` as a decision tool; ``turns`` is ``{idx: (state, live choice)}``."""
    store.create_session(
        {
            "id": sid,
            "example": "multilingual-assistant",
            "started_at": started,
            "last_seen_at": ended,
            "ended_at": ended,
            "config": {
                "language": "fr-FR",
                "decision_tools": {
                    "model": "d3:2b",
                    "tools": ["play_emotion"],
                    "questions": {"play_emotion": EMOTION_QUESTION},
                },
            },
            "artifact_prefix": f"sessions/{sid}/",
        }
    )
    rows = []
    for idx, (state, choice) in turns.items():
        data = {"outcome": "called", "tool": "play_emotion", "choice": choice, "state": state}
        rows.append(
            ("events", {"session_id": sid, "turn_idx": idx, "ts": started + idx, "kind": "decision_tool", "data": data})
        )
        # A later answer of the same turn is not replayed.
        later = {**data, "state": f"{state} (later)", "choice": "sad"}
        rows.append(
            (
                "events",
                {"session_id": sid, "turn_idx": idx, "ts": started + idx + 0.5, "kind": "decision_tool", "data": later},
            )
        )
    store.write_batch(rows)


def test_redecide_scores_live_and_candidates(stores, monkeypatch):
    import httpx

    from monitoring.jobs import redecide as redecide_module

    store, _ = stores
    clock = FakeClock()
    _decision_session(
        store,
        "s1",
        started=clock.now - 500,
        ended=clock.now - 400,
        turns={
            1: ("User: J'ai réussi !\nAssistant: Bravo !", "happy"),
            2: ("User: Mon chat est mort.\nAssistant: Désolé.", "happy"),
        },
    )
    _session(store, "s2", started=clock.now - 300, ended=clock.now - 200)  # no decision tools
    asked: list[dict] = []
    picks = {"nimble": {"J'ai": "happy", "chat": "sad"}, "d3:9b": {"J'ai": "happy", "chat": "happy"}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body)
        word = "chat" if "chat" in body["state"] else "J'ai"
        choice = picks[body["model"]][word]
        answer = {"type": "choice", "choice": choice, "confidence": 0.9, "probabilities": {choice: 0.9}}
        return httpx.Response(200, json={"answers": {"play_emotion": answer}})

    real_client = httpx.Client
    monkeypatch.setattr(
        redecide_module.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setenv("DECISION_MODEL_URL", "http://ollama.test:11434/v1/systemone")
    config = {
        "auto_enqueue": ["redecide"],
        "redecide": {
            "reference": {"name": "nimble", "model": "nimble"},
            "candidates": [{"name": "d3-9b", "model": "d3:9b"}, {"name": "off", "model": "x", "enabled": False}],
        },
    }
    dreamer = _dreamer(stores, clock, **config)
    clock.now += 1000
    while dreamer.run_once():
        pass

    # Two turns x two models, each with the session's question and the first answer's state only.
    assert len(asked) == 4
    assert all(body["questions"] == {"play_emotion": EMOTION_QUESTION} for body in asked)
    assert not any("(later)" in body["state"] for body in asked)
    summaries = {a["source"]: a["value"] for a in store.annotations_for("s1", kind="decision_summary")}
    assert summaries["live:d3:2b"] == {"agreement": 0.5, "agreed": 1, "turns": 2, "reference_sources": ["nimble"]}
    assert summaries["d3-9b"]["agreed"] == 1
    assert store.annotations_for("s2", kind="decision") == []

    # A re-run only fills gaps; a human choice becomes the reference.
    store.add_annotations(
        [
            {
                "session_id": "s1",
                "target_type": "turn",
                "target_id": "s1:2",
                "source": "human:qa",
                "kind": "decision",
                "value": {"answers": {"play_emotion": {"choice": "happy"}}},
            }
        ]
    )
    store.reset_job("redecide", "s1")
    while dreamer.run_once():
        pass
    assert len(asked) == 4
    latest = {a["source"]: a["value"] for a in store.annotations_for("s1", kind="decision_summary")}
    assert latest["live:d3:2b"]["agreed"] == 2
    assert latest["live:d3:2b"]["reference_sources"] == ["human:qa", "nimble"]


def test_redecide_endpoints_need_a_url(monkeypatch):
    from monitoring.jobs.redecide import redecide_endpoints

    config = {"redecide": {"reference": {"name": "nimble", "model": "nimble"}}}
    monkeypatch.delenv("DECISION_MODEL_URL", raising=False)
    assert redecide_endpoints(config) == (None, [])
    config["redecide"]["reference"]["url"] = "http://other:11434/v1/systemone"
    reference, _ = redecide_endpoints(config)
    assert (reference.name, reference.url) == ("nimble", "http://other:11434/v1/systemone")
    assert "redecide" in JOB_REGISTRY


def test_openai_transcriber_sends_iso_language(tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from monitoring.jobs.asr_client import AsrEndpoint, OpenAITranscriber, make_transcriber

    received = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json(
                {"object": "list", "data": [{"id": "voxtral", "object": "model", "created": 0, "owned_by": "x"}]}
            )

        def do_POST(self):
            received["path"] = self.path
            received["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self._json({"text": " Allume la lumière. "})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        endpoint = AsrEndpoint.from_config(
            {"name": "voxtral", "protocol": "openai", "base_url": f"http://127.0.0.1:{server.server_port}/v1"}
        )
        transcriber = make_transcriber(endpoint)
        assert isinstance(transcriber, OpenAITranscriber)
        transcriber.wait_ready(timeout_secs=5)
        text = transcriber.transcribe_wav(pcm16_to_wav(b"\x00\x00" * 1600, 16000), "fr-FR")
    finally:
        server.shutdown()
    assert text == "Allume la lumière."
    assert received["path"] == "/v1/audio/transcriptions"
    assert b'name="language"\r\n\r\nfr\r\n' in received["body"]
    assert b'name="model"\r\n\r\nvoxtral\r\n' in received["body"]


def test_pause_blocks_and_preempts(stores):
    store, _ = stores
    clock = FakeClock()
    _session(store, "s1", started=clock.now - 500, ended=clock.now - 400)
    dreamer = _dreamer(stores, clock)

    store.set_kv("dreamer.paused", {"paused": True})
    assert not dreamer.run_once()
    assert store.get_kv("dreamer.status")["value"]["state"] == "paused"

    store.set_kv("dreamer.paused", {"paused": False})

    def on_step(step):
        if step == 0:
            store.set_kv("dreamer.paused", {"paused": True})
            clock.now += 2

    StepJob.on_step = on_step
    try:
        assert dreamer.run_once()
    finally:
        StepJob.on_step = None
    (job,) = store.jobs()
    assert job["status"] == "pending" and job["progress"] == {"next": 1}
    status = store.get_kv("dreamer.status")["value"]
    assert status["heartbeat_at"] == clock.now and "jobs" in status


def test_disabled_reasr_endpoints_are_skipped():
    from monitoring.jobs.reasr import reasr_endpoints
    from monitoring.jobs.runner import declared_services, load_dreamer_config

    config = {
        "reasr": {
            "reference": {"name": "voxtral", "protocol": "openai", "service": "voxtral-svc"},
            "candidates": [
                {"name": "on", "server": "a:1", "service": "on-svc"},
                {"name": "off", "server": "b:1", "service": "off-svc", "enabled": False},
            ],
        }
    }
    reference, candidates = reasr_endpoints(config)
    assert reference.name == "voxtral" and [c.name for c in candidates] == ["on"]
    assert declared_services(config) == ["on-svc", "voxtral-svc"]

    config["reasr"]["reference"]["enabled"] = False
    assert reasr_endpoints(config)[0] is None
    assert declared_services(config) == ["on-svc"]

    # Shipped default: only the Voxtral reference container is started.
    assert declared_services(load_dreamer_config()) == ["dreamer-asr-voxtral"]

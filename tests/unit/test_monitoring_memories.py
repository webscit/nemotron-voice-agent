# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from monitoring import memories
from monitoring.api import create_review_router
from monitoring.config import MonitoringConfig
from monitoring.jobs import JobContext
from monitoring.jobs.dream import DreamJob, parse_memories
from monitoring.jobs.runner import _DEFAULTS
from monitoring.store import LocalArtifactStore, SessionStore


@pytest.fixture
def stores(tmp_path):
    store = SessionStore(f"sqlite:///{tmp_path / 'db.sqlite'}")
    store.create_schema()
    return store, LocalArtifactStore(tmp_path / "artifacts")


class FakeLlm:
    """OpenAI-compatible server returning a canned chat answer."""

    def __init__(self, answer):
        self.answer = answer
        self.requests: list[dict] = []
        outer = self

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
                    {"object": "list", "data": [{"id": "llm", "object": "model", "created": 0, "owned_by": "x"}]}
                )

            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                content = outer.answer if isinstance(outer.answer, str) else json.dumps(outer.answer)
                self._json(
                    {
                        "id": "c",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "llm",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": content},
                            }
                        ],
                    }
                )

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def llm():
    fake = FakeLlm({"memories": []})
    yield fake
    fake.close()


def _session(store, sid, base_url, *, started=1000.0):
    store.create_session(
        {
            "id": sid,
            "example": "multilingual-assistant",
            "started_at": started,
            "ended_at": started + 60,
            "config": {
                "language": "fr-FR",
                "asr": {"model": "live-asr"},
                "llm": {
                    "model": "llm",
                    "base_url": base_url,
                    "extra": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
                },
            },
            "artifact_prefix": f"sessions/{sid}/",
        }
    )
    store.write_batch(
        [
            ("turns", {"session_id": sid, "idx": 0, "bot_text": "Bonjour !"}),
            ("turns", {"session_id": sid, "idx": 1, "user_text": "je suis vegetarien", "bot_text": "Noté."}),
            ("turns", {"session_id": sid, "idx": 2, "user_text": "ma fille s'appelle Léa", "bot_text": "Joli."}),
        ]
    )


def _run_dream(store, artifacts, sid, config=None):
    store.reset_job("dream", sid)
    job = next(j for j in store.jobs() if j["kind"] == "dream" and j["target"] == sid)
    ctx = JobContext(
        job=job,
        store=store,
        artifacts=artifacts,
        config={**_DEFAULTS, **(config or {})},
        is_preempted=lambda: False,
        progress=dict(job["progress"] or {}),
    )
    DreamJob().run(ctx)
    return ctx.progress


def _memory(person_id, text, turns, confidence=0.95, supersedes=None):
    return {
        "person_id": person_id,
        "text": text,
        "category": "preference",
        "confidence": confidence,
        "turns": turns,
        "quote": "q",
        "supersedes": supersedes,
    }


def test_parse_memories_tolerates_think_tags_and_prose():
    raw = '<think>hmm {"no": 1}</think>Here: {"memories": [{"text": "a"}, "junk"]} done'
    assert parse_memories(raw) == [{"text": "a"}]
    with pytest.raises(ValueError):
        parse_memories("nothing here")


def test_dream_skips_sessions_without_speaker(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    assert _run_dream(store, artifacts, "s1") == {"skipped": "no speaker"}
    assert llm.requests == []


def test_dream_threshold_evidence_and_transcript_source(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    alice = memories.create_person(store, "Alice")
    bob = memories.create_person(store, "Bob")
    memories.assign_speaker(store, "s1", alice["id"], source="live:picker")
    memories.assign_speaker(store, "s1", bob["id"], source="human:rev", turn_idx=2)
    store.add_annotations(
        [
            {
                "session_id": "s1",
                "target_type": "turn",
                "target_id": "s1:1",
                "source": "human:rev",
                "kind": "transcript",
                "value": {"text": "je suis végétarienne"},
            }
        ]
    )
    llm.answer = {
        "memories": [
            _memory(alice["id"], "Alice est végétarienne.", [1]),
            _memory(bob["id"], "Bob a une fille, Léa.", [2], confidence=0.6),
            _memory(alice["id"], "Alice a une fille.", [2]),  # turn 2 is Bob's: dropped
            _memory("stranger", "Nope.", [1]),
        ]
    }
    progress = _run_dream(store, artifacts, "s1")

    assert progress["created"] == 2 and progress["active"] == 1
    rows = memories.list_memories(store, status="all")["memories"]
    by_text = {m["text"]: m for m in rows}
    assert by_text["Alice est végétarienne."]["status"] == "active"
    assert by_text["Bob a une fille, Léa."]["status"] == "proposed"
    assert by_text["Bob a une fille, Léa."]["evidence"][0]["turn_idx"] == 2
    request = llm.requests[0]
    assert request["chat_template_kwargs"]["enable_thinking"] is True
    prompt = request["messages"][1]["content"]
    assert "[turn 1] Alice: je suis végétarienne" in prompt  # human reference beats live ASR
    assert "[turn 2] Bob: ma fille s'appelle Léa" in prompt
    assert "French" in request["messages"][0]["content"]


def test_dream_supersedes_and_never_touches_reviewed(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    alice = memories.create_person(store, "Alice")
    memories.assign_speaker(store, "s1", alice["id"], source="live:picker")
    old_auto = memories.add_memory(
        store, {"person_id": alice["id"], "text": "Alice vit à Lyon.", "status": "active", "source": "dream:x"}, []
    )
    reviewed = memories.add_memory(
        store,
        {"person_id": alice["id"], "text": "Alice aime le thé.", "status": "active", "source": "dream:x"},
        [],
    )
    memories.review_memory(store, reviewed, "approve", reviewer="rev")
    llm.answer = {
        "memories": [
            _memory(alice["id"], "Alice vit à Paris.", [1], supersedes=old_auto),
            _memory(alice["id"], "Alice préfère le café.", [2], supersedes=reviewed),
        ]
    }
    _run_dream(store, artifacts, "s1")

    assert memories.get_memory(store, old_auto)["status"] == "superseded"
    assert memories.get_memory(store, reviewed)["status"] == "active"
    rows = {m["text"]: m for m in memories.list_memories(store, status="all")["memories"]}
    assert rows["Alice vit à Paris."]["status"] == "active"
    coffee = rows["Alice préfère le café."]
    assert coffee["status"] == "proposed" and coffee["supersedes"] == reviewed
    assert coffee["supersedes_text"] == "Alice aime le thé."

    # Approving the proposal replaces the reviewed memory.
    memories.review_memory(store, coffee["id"], "approve", reviewer="rev")
    assert memories.get_memory(store, reviewed)["status"] == "superseded"


def test_dream_rerun_retracts_unreviewed_memories_of_the_session(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    alice = memories.create_person(store, "Alice")
    memories.assign_speaker(store, "s1", alice["id"], source="live:picker")
    llm.answer = {"memories": [_memory(alice["id"], "Alice est végétarienne.", [1])]}
    _run_dream(store, artifacts, "s1")
    first = memories.usable_memories(store, [alice["id"]])
    assert len(first) == 1

    llm.answer = {"memories": [_memory(alice["id"], "Alice est végane.", [1])]}
    progress = _run_dream(store, artifacts, "s1")
    assert progress["retracted"] == 1
    assert [m["text"] for m in memories.usable_memories(store, [alice["id"]])] == ["Alice est végane."]
    assert memories.get_memory(store, first[0]["id"])["status"] == "superseded"


def test_retract_keeps_memories_backed_by_other_sessions(stores):
    store, _ = stores
    alice = memories.create_person(store, "Alice")
    shared = memories.add_memory(
        store,
        {"person_id": alice["id"], "text": "Alice a un chien.", "status": "active", "source": "dream:x"},
        [{"session_id": "s1", "turn_idx": 1, "quote": None}, {"session_id": "s2", "turn_idx": 3, "quote": None}],
    )
    only_s1 = memories.add_memory(
        store,
        {"person_id": alice["id"], "text": "Alice joue du piano.", "status": "proposed", "source": "dream:x"},
        [{"session_id": "s1", "turn_idx": 2, "quote": None}],
    )
    assert memories.retract_session_memories(store, "s1") == 1
    assert memories.get_memory(store, shared)["status"] == "active"
    assert memories.get_memory(store, only_s1)["status"] == "superseded"


def test_review_correct_and_forget(stores):
    store, _ = stores
    alice = memories.create_person(store, "Alice")
    bob = memories.create_person(store, "Bob")
    mid = memories.add_memory(
        store,
        {"person_id": alice["id"], "text": "Alice a un chat.", "status": "proposed", "source": "dream:x"},
        [{"session_id": "s1", "turn_idx": 1, "quote": "mon chat"}],
    )
    corrected = memories.review_memory(
        store, mid, "correct", reviewer="rev", text="Bob a un chat.", person_id=bob["id"]
    )
    assert corrected["person_id"] == bob["id"] and corrected["status"] == "active"
    assert corrected["source"] == "human:rev" and corrected["supersedes"] == mid
    assert memories.get_memory(store, mid)["superseded_by"] == corrected["id"]
    assert memories.list_memories(store, status="all")["memories"][0]["evidence"][0]["quote"] == "mon chat"

    forgotten = memories.review_memory(store, corrected["id"], "forget", reviewer="rev")
    assert forgotten["status"] == "forgotten"
    assert memories.active_memories(store, bob["id"]) == []
    assert memories.open_memory_count(store) == 0


@pytest.fixture
def api(tmp_path):
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
    _session(store, "s1", "http://unused/v1")
    app = FastAPI()
    app.include_router(create_review_router(settings, store, artifacts, {**_DEFAULTS, "reasr": {}}))
    return TestClient(app), store


def test_api_people_speakers_and_memories(api):
    client, store = api
    person = client.post("/api/review/people", json={"name": " Alice "}).json()
    assert person["name"] == "Alice"
    assert client.patch(f"/api/review/people/{person['id']}", json={"name": "Alicia"}).json()["name"] == "Alicia"
    assert client.patch("/api/review/people/nope", json={"name": "X"}).status_code == 404

    assign = client.post("/api/review/sessions/s1/speaker", json={"annotator": "rev", "person_id": person["id"]})
    assert assign.json() == {"session": person["id"], "turns": {}}
    assert any(j["kind"] == "dream" and j["status"] == "pending" for j in store.jobs())
    bad = client.post("/api/review/sessions/s1/speaker", json={"annotator": "rev", "person_id": "nope"})
    assert bad.status_code == 422
    client.post("/api/review/sessions/s1/speaker", json={"annotator": "rev", "person_id": None, "turn_idx": 2})

    listed = client.get("/api/review/sessions").json()["sessions"][0]
    assert listed["person"] == {"id": person["id"], "name": "Alicia"}

    people = client.get("/api/review/people").json()["people"]
    assert people[0]["sessions"] == 1

    mid = memories.add_memory(
        store,
        {"person_id": person["id"], "text": "Alicia a un chat.", "status": "active", "source": "dream:x"},
        [{"session_id": "s1", "turn_idx": 1, "quote": "chat"}],
    )
    memories.record_uses(store, "s1", [mid])
    activities = {a["id"]: a for a in client.get("/api/review/activities").json()["activities"]}
    assert activities["memories"]["open"] == 1

    page = client.get("/api/review/memories").json()
    item = page["memories"][0]
    assert page["total"] == 1 and item["person_name"] == "Alicia"
    assert item["evidence"][0]["user_text"] == "je suis vegetarien"
    assert item["used_in_sessions"] == 1 and item["used_in_replies"] == 3

    detail = client.get("/api/review/sessions/s1").json()
    assert detail["speakers"] == {"session": person["id"]}
    assert [m["id"] for m in detail["memories_used"]["used"]] == [mid]
    assert [m["id"] for m in detail["memories_used"]["extracted"]] == [mid]

    reviewed = client.post(f"/api/review/memories/{mid}/review", json={"annotator": "rev", "action": "approve"})
    assert reviewed.json()["reviewed_by"] == "human:rev"
    assert client.get("/api/review/memories").json()["total"] == 0
    empty_fix = client.post(f"/api/review/memories/{mid}/review", json={"annotator": "rev", "action": "correct"})
    assert empty_fix.status_code == 422
    assert (
        client.post("/api/review/memories/999/review", json={"annotator": "rev", "action": "forget"}).status_code == 404
    )


def test_person_context_prompt_and_errors(stores, monkeypatch):
    from examples.shared import person_memory

    store, _ = stores
    alice = memories.create_person(store, "Alice")
    mid = memories.add_memory(
        store, {"person_id": alice["id"], "text": "Alice aime le thé.", "status": "active", "source": "x"}, []
    )
    memories.add_memory(
        store, {"person_id": alice["id"], "text": "Brouillon.", "status": "proposed", "source": "x"}, []
    )
    monkeypatch.setattr(memories, "live_store", lambda: store)

    context = asyncio.run(person_memory.load_person_context(alice["id"]))
    assert context.snapshot == {"id": alice["id"], "name": "Alice"}
    assert context.prompt_replacements() == {"person_name": "Alice", "memories": "- Alice aime le thé."}
    assert asyncio.run(person_memory.load_person_context("unknown")) is None
    assert asyncio.run(person_memory.load_person_context(None)) is None

    asyncio.run(person_memory.record_person_session("s9", context))
    assert memories.speakers_for(store, "s9") == {"session": alice["id"], "turns": {}}
    asyncio.run(person_memory.record_person_session("s9", context))  # idempotent uses

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(memories, "live_store", boom)
    assert asyncio.run(person_memory.load_person_context(alice["id"])) is None
    asyncio.run(person_memory.record_person_session("s9", context))  # swallowed
    assert mid

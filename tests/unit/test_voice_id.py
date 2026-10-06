# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.shared import voice_id as live
from examples.shared.voice_id import (
    ENROLL_TOOL_NAME,
    GUEST,
    Speaker,
    VoiceIdSession,
    VoiceIdTracker,
    parse_presence_update,
    parse_speaker_update,
    speaker_label,
    tag_user_message,
    with_enroll_tool,
)
from monitoring import memories, voice_id
from monitoring.store import SessionStore

MODEL = "test-model@1"
FACE_MODEL = "test-face@1"
DIM = 32


def _vec(seed: int) -> np.ndarray:
    return voice_id.normalize(np.random.default_rng(seed).normal(size=DIM))


def _payload(**overrides) -> dict:
    payload = {
        "model": MODEL,
        "utterance_id": 1,
        "person_id": None,
        "provisional_id": None,
        "score": 0.7,
        "tier": "none",
        "speech_ms": 1500,
        "final": False,
    }
    payload.update(overrides)
    return payload


def _update(now: float = 0.0, **overrides):
    update = parse_speaker_update(_payload(**overrides), now=now)
    assert update is not None
    return update


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = SessionStore(f"sqlite:///{tmp_path / 'db.sqlite'}")
    store.create_schema()
    monkeypatch.setattr(memories, "live_store", lambda: store)
    return store


# ------------------------------------------------------------------ storage
def test_vector_wire_format_round_trip_and_rejects_garbage():
    vector = _vec(1)
    decoded = voice_id.decode_vector(voice_id.encode_vector(vector * 3))
    assert np.allclose(decoded, vector, atol=1e-6)
    assert abs(float(np.linalg.norm(decoded)) - 1.0) < 1e-5
    assert voice_id.decode_vector("not base64!") is None
    assert voice_id.decode_vector(voice_id.encode_vector(np.zeros(DIM))) is None
    assert voice_id.decode_vector(voice_id.encode_vector(np.ones(4))) is None  # too short to be an embedding
    assert voice_id.decode_vector(voice_id.encode_vector(np.full(DIM, np.nan))) is None
    assert voice_id.decode_vector(None) is None


def test_find_or_create_person_is_case_insensitive_and_skips_archived(store):
    elodie = memories.find_or_create_person(store, " Élodie ")
    assert memories.find_or_create_person(store, "élodie")["id"] == elodie["id"]
    memories.update_person(store, elodie["id"], archived=True)
    assert memories.find_person_by_name(store, "Élodie") is None
    assert memories.find_or_create_person(store, "Élodie")["id"] != elodie["id"]


def test_centroids_are_per_model_normalised_and_ignore_other_dimensions(store):
    alice = memories.create_person(store, "Alice")
    bob = memories.create_person(store, "Bob")
    a1, a2 = _vec(1), _vec(2)
    assert voice_id.add_embeddings(store, alice["id"], MODEL, [a1, a2, np.zeros(DIM)], source="t") == 2
    voice_id.add_embeddings(store, alice["id"], MODEL, [np.ones(DIM * 2)], source="t")  # another dimension
    voice_id.add_embeddings(store, bob["id"], "other-model@1", [_vec(3)], source="t")

    found = voice_id.centroids(store, MODEL)
    assert set(found) == {alice["id"]}
    assert found[alice["id"]]["count"] == 2
    assert np.allclose(found[alice["id"]]["centroid"], voice_id.normalize(a1 + a2), atol=1e-6)
    assert voice_id.centroids(store, "missing@1") == {}

    assert voice_id.delete_embeddings(store, alice["id"], model=MODEL) == 3
    assert voice_id.centroids(store, MODEL) == {}


def test_schema_upgrade_adds_table_to_existing_store(tmp_path):
    from sqlalchemy import inspect

    from monitoring import schema

    url = f"sqlite:///{tmp_path / 'old.sqlite'}"
    old = SessionStore(url)
    tables = [t for t in schema.metadata.sorted_tables if t.name != "voice_embeddings"]
    schema.metadata.create_all(old.engine, tables=tables)
    assert "voice_embeddings" not in inspect(old.engine).get_table_names()
    upgraded = SessionStore(url)
    upgraded.create_schema()
    assert "voice_embeddings" in inspect(upgraded.engine).get_table_names()


# ------------------------------------------------------------------ gallery
def test_gallery_endpoint(store):
    app = FastAPI()
    app.include_router(voice_id.create_voice_id_router())
    client = TestClient(app)
    alice = memories.create_person(store, "Alice")
    archived = memories.create_person(store, "Gone")
    memories.create_person(store, "No voice")
    voice_id.add_embeddings(store, alice["id"], MODEL, [_vec(1)], source="t")
    voice_id.add_embeddings(store, archived["id"], MODEL, [_vec(2)], source="t")
    memories.update_person(store, archived["id"], archived=True)

    body = client.get("/api/voice-id/gallery", params={"model": MODEL}).json()
    assert body["model"] == MODEL
    assert [(p["person_id"], p["name"], p["count"]) for p in body["people"]] == [(alice["id"], "Alice", 1)]
    assert np.allclose(voice_id.decode_vector(body["people"][0]["centroid"]), _vec(1), atol=1e-6)
    assert client.get("/api/voice-id/gallery", params={"model": "other@1"}).json()["people"] == []
    assert client.get("/api/voice-id/gallery").status_code == 422

    deleted = client.delete(f"/api/voice-id/people/{alice['id']}/embeddings").json()
    assert deleted == {"person_id": alice["id"], "deleted": 1}


def test_gallery_endpoint_is_503_without_people_store(monkeypatch):
    app = FastAPI()
    app.include_router(voice_id.create_voice_id_router())
    client = TestClient(app)
    monkeypatch.setattr(memories, "live_store", lambda: None)
    assert client.get("/api/voice-id/gallery", params={"model": MODEL}).status_code == 503

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(memories, "live_store", boom)
    assert client.get("/api/voice-id/gallery", params={"model": MODEL}).status_code == 503


# ------------------------------------------------------------------ parsing
def test_parse_speaker_update_validates_and_normalises_tiers():
    assert parse_speaker_update("nope") is None
    assert parse_speaker_update(_payload(model="")) is None
    assert parse_speaker_update(_payload(model="x" * 200)) is None

    match = _update(person_id="p1", provisional_id="unk-1", tier="high", score=7, final=True)
    assert (match.person_id, match.provisional_id, match.tier, match.score) == ("p1", None, "high", 1.0)
    assert match.embedding is None  # final without an embedding is fine

    assert _update(tier="high").tier == "none"  # a match needs a person
    assert _update(tier="low", provisional_id="unk-2").tier == "unknown"
    unknown = _update(tier="unknown", person_id="p1", provisional_id="unk-2")
    assert (unknown.person_id, unknown.provisional_id, unknown.tier) == (None, "unk-2", "unknown")
    assert _update(tier="unknown").tier == "none"  # unknown needs a provisional id
    assert _update(tier="bogus", person_id="p1").tier == "none"
    assert _update(utterance_id="x", speech_ms=-5, score=float("nan")).speech_ms == 0

    running = _update(provisional_id="unk-1", tier="unknown", embedding=voice_id.encode_vector(_vec(1)))
    assert running.embedding is None  # embeddings only count on final updates
    final = _update(provisional_id="unk-1", tier="unknown", final=True, embedding=voice_id.encode_vector(_vec(1)))
    assert np.allclose(final.embedding, _vec(1), atol=1e-6)
    assert _update(provisional_id="unk-1", tier="unknown", final=True, embedding="@@").embedding is None


# ------------------------------------------------------------------ tracker
def test_tracker_uses_fresh_update_and_falls_back_to_previous_speaker_at_low():
    tracker = VoiceIdTracker()
    assert tracker.current_speaker() is GUEST and GUEST.is_guest

    tracker.start_turn(now=10.0)
    tracker.on_update(_update(now=11.0, person_id="p1", tier="high"))
    speaker = tracker.commit_turn(now=12.0)
    assert (speaker.person_id, speaker.tier, speaker.fallback) == ("p1", "high", False)
    assert speaker.trusted_person_id == "p1" and not speaker.is_guest

    # Short turn without any update: previous speaker, reduced confidence, guest policies.
    tracker.start_turn(now=20.0)
    speaker = tracker.commit_turn(now=20.5)
    assert (speaker.person_id, speaker.tier, speaker.fallback) == ("p1", "low", True)
    assert speaker.is_guest

    # Tier ``none`` ("too little speech") falls back as well.
    tracker.start_turn(now=30.0)
    tracker.on_update(_update(now=30.2, tier="none"))
    assert tracker.commit_turn(now=30.5).person_id == "p1"

    # A stale update from long before the turn is not reused.
    tracker.on_update(_update(now=31.0, person_id="p2", tier="high"))
    tracker.start_turn(now=40.0)
    assert tracker.commit_turn(now=41.0).person_id == "p1"

    # ... but one that arrived just before the server noticed the turn is.
    tracker.on_update(_update(now=49.5, person_id="p2", tier="low", utterance_id=7))
    tracker.start_turn(now=50.0)
    speaker = tracker.commit_turn(now=51.0)
    assert (speaker.person_id, speaker.tier, speaker.is_guest) == ("p2", "low", True)

    # The utterance that decided a turn does not decide the next one, however close.
    tracker.start_turn(now=51.2)
    speaker = tracker.commit_turn(now=51.5)
    assert (speaker.person_id, speaker.fallback) == ("p2", True)


def test_tracker_unknown_speaker_stays_unknown_on_fallback_and_nobody_is_guest():
    tracker = VoiceIdTracker()
    tracker.start_turn(now=1.0)
    assert tracker.commit_turn(now=2.0) == Speaker(fallback=True)

    tracker.start_turn(now=10.0)
    tracker.on_update(_update(now=11.0, provisional_id="unk-1", tier="unknown"))
    assert tracker.commit_turn(now=12.0).provisional_id == "unk-1"
    tracker.start_turn(now=20.0)
    speaker = tracker.commit_turn(now=21.0)
    assert (speaker.provisional_id, speaker.tier, speaker.fallback) == ("unk-1", "unknown", True)


def test_tracker_seeded_with_picked_person():
    tracker = VoiceIdTracker("p1")
    assert tracker.current_speaker().trusted_person_id == "p1"
    assert not tracker.active


def test_tracker_late_final_update_corrects_a_fallback_turn_once():
    tracker = VoiceIdTracker()
    tracker.start_turn(now=1.0)
    tracker.commit_turn(now=2.0)
    assert tracker.late_correction(_update(now=2.5, person_id="p1", tier="high")) is None  # not final
    late = _update(now=3.0, person_id="p1", tier="high", final=True)
    assert tracker.late_correction(late).person_id == "p1"
    assert tracker.late_correction(late) is None  # no longer a fallback
    tracker.start_turn(now=10.0)
    tracker.commit_turn(now=11.0)
    assert tracker.late_correction(_update(now=60.0, person_id="p2", tier="high", final=True)) is None  # too late


def test_tracker_buffers_final_embeddings_and_relabels_on_enrollment():
    tracker = VoiceIdTracker()
    tracker.start_turn(now=1.0)
    for seed in range(live.MAX_BUFFERED_EMBEDDINGS + 5):
        tracker.on_update(
            _update(
                now=1.5,
                provisional_id="unk-1",
                tier="unknown",
                final=True,
                embedding=voice_id.encode_vector(_vec(seed)),
            )
        )
    tracker.on_update(_update(now=1.6, provisional_id="unk-1", tier="unknown"))  # running: not buffered
    tracker.commit_turn(now=2.0)
    assert len(tracker.enrollment_sample()) == live.MAX_BUFFERED_EMBEDDINGS

    assert tracker.mark_enrolled("p9") == "unk-1"
    speaker = tracker.current_speaker()
    assert (speaker.person_id, speaker.provisional_id, speaker.tier) == ("p9", None, "high")
    assert tracker.enrollment_sample() == []  # consumed

    # Until the client relabels its cluster, the provisional id means that person (not trusted yet).
    tracker.start_turn(now=10.0)
    tracker.on_update(_update(now=10.5, provisional_id="unk-1", tier="unknown"))
    speaker = tracker.commit_turn(now=11.0)
    assert (speaker.person_id, speaker.tier) == ("p9", "low")


def test_tracker_latest_turn_only_sample():
    tracker = VoiceIdTracker()
    for turn, seed in ((1.0, 1), (10.0, 2)):
        tracker.start_turn(now=turn)
        tracker.on_update(
            _update(
                now=turn + 0.5, person_id="p1", tier="low", final=True, embedding=voice_id.encode_vector(_vec(seed))
            )  # noqa: E501
        )
        tracker.commit_turn(now=turn + 1.0)
    assert len(tracker.enrollment_sample()) == 2
    latest = tracker.enrollment_sample(latest_turn_only=True)
    assert len(latest) == 1 and np.allclose(latest[0], _vec(2), atol=1e-6)


# ------------------------------------------------------------- LLM surface
def test_speaker_tags():
    assert speaker_label(Speaker(person_id="p", tier="high"), "Alice") == "Alice"
    assert speaker_label(Speaker(person_id="p", tier="low"), "Alice") == "Alice (uncertain)"
    assert speaker_label(Speaker(provisional_id="unk-1", tier="unknown"), None) == "unknown guest"

    message = {"role": "user", "content": "Bonjour"}
    assert tag_user_message(message, "Alice")
    assert message["content"] == "[speaker: Alice] Bonjour"
    assert not tag_user_message(message, "Bob")  # tagged once
    parts = {"role": "user", "content": [{"type": "text", "text": "Bonjour"}]}
    assert tag_user_message(parts, "Alice") and parts["content"][0]["text"] == "[speaker: Alice]"
    assert not tag_user_message(parts, "Alice")
    assert not tag_user_message({"role": "assistant", "content": "x"}, "Alice")


def test_with_enroll_tool_keeps_client_tools_and_is_idempotent():
    assert [t.name for t in with_enroll_tool(LLMContext([]).tools).standard_tools] == [ENROLL_TOOL_NAME]
    custom = {AdapterType.OPENAI: [{"type": "function", "function": {"name": "dance"}}]}
    tools = with_enroll_tool(with_enroll_tool(ToolsSchema(standard_tools=[], custom_tools=custom)))
    assert [t.name for t in tools.standard_tools] == [ENROLL_TOOL_NAME]
    assert tools.custom_tools == custom


# ------------------------------------------------------------------ session
class _Harness:
    def __init__(self, store, *, initial_person=None, system_prompt=""):
        self.frames: list = []
        self.turn_idx = 0
        messages = [{"role": "system", "content": "control"}] if system_prompt else []
        messages.append({"role": "user" if system_prompt else "system", "content": "BASE"})
        self.context = LLMContext(messages)
        self.session = VoiceIdSession(
            context=self.context,
            pinned_index=len(messages) - 1,
            render_pinned=self._render,
            queue_frame=self._queue,
            session_id=lambda: "s1",
            turn_idx=lambda: self.turn_idx,
            initial_person=initial_person,
        )
        self.pinned = messages[-1]

    @staticmethod
    def _render(person, voice_id_active=False, face_id_active=False):
        content = "BASE"
        if voice_id_active:
            content += "|VOICE"
        if face_id_active:
            content += "|FACE"
        if person:
            content += f"|{person.person['name']}:{person.prompt_replacements()['memories']}"
        return content

    async def _queue(self, frame):
        self.frames.append(frame)

    async def turn(self, text, **update):
        self.turn_idx += 1
        self.session.on_user_turn_started()
        if update:
            await self.session.on_speaker_update(_payload(**update))
        self.context.add_message({"role": "user", "content": text})
        await self.session.commit_turn(self.context)
        await self.drain()
        return self.context.get_messages()[-1]["content"]

    async def drain(self):
        while self.session._background:
            await asyncio.gather(*list(self.session._background))


def _memory(store, person, text):
    memories.add_memory(store, {"person_id": person["id"], "text": text, "status": "active", "source": "x"}, [])


def test_session_is_inert_until_the_client_sends_updates(store):
    async def run():
        h = _Harness(store)
        assert await h.turn("Bonjour") == "Bonjour"
        assert h.pinned["content"] == "BASE"
        assert not isinstance(h.context.tools, ToolsSchema)
        assert memories.speakers_for(store, "s1") == {"session": None, "turns": {}}
        await h.session.on_speaker_update({"model": ""})  # malformed: still inert
        assert not h.session.tracker.active

    asyncio.run(run())


def test_session_tags_turns_swaps_memories_and_records_speakers(store):
    alice = memories.create_person(store, "Alice")
    bob = memories.create_person(store, "Bob")
    _memory(store, alice, "Alice aime le thé.")
    _memory(store, bob, "Bob a un chat.")

    async def run():
        h = _Harness(store, system_prompt="control")
        assert await h.turn("Bonjour", person_id=alice["id"], tier="high") == "[speaker: Alice] Bonjour"
        assert h.pinned["content"] == "BASE|VOICE|Alice:- Alice aime le thé."
        assert h.context.get_messages()[0] == {"role": "system", "content": "control"}
        assert [t.name for t in h.context.tools.standard_tools] == [ENROLL_TOOL_NAME]
        assert h.session.current_speaker().trusted_person_id == alice["id"]

        # Short turn without an update: same person, uncertain, memories kept.
        assert await h.turn("Oui") == "[speaker: Alice (uncertain)] Oui"
        assert "Alice aime le thé" in h.pinned["content"]
        assert h.session.current_speaker().is_guest

        # Somebody else, confidently: their memories replace Alice's.
        assert await h.turn("Salut", person_id=bob["id"], tier="high") == "[speaker: Bob] Salut"
        assert h.pinned["content"] == "BASE|VOICE|Bob:- Bob a un chat."

        # An uncertain match of another person must not expose that person's memories.
        assert await h.turn("Hm", person_id=alice["id"], tier="low") == "[speaker: Alice (uncertain)] Hm"
        assert h.pinned["content"] == "BASE|VOICE"

        # Unknown voice, then an id the store does not know (stale client gallery).
        assert await h.turn("Hello", provisional_id="unk-1", tier="unknown") == "[speaker: unknown guest] Hello"
        assert await h.turn("Hey", person_id="ghost", tier="high") == "[speaker: unknown guest] Hey"
        assert h.session.current_speaker().is_guest

    asyncio.run(run())

    speakers = memories.speakers_for(store, "s1")
    assert speakers["turns"] == {1: alice["id"], 2: alice["id"], 3: bob["id"], 4: alice["id"]}
    from sqlalchemy import select

    from monitoring import schema

    with store.engine.connect() as conn:
        sa = schema.speaker_assignments
        sources = dict(conn.execute(select(sa.c.turn_idx, sa.c.source)).all())
        used = {r[0] for r in conn.execute(select(schema.memory_uses.c.session_id))}
    assert sources[1] == f"live:voice-id:{MODEL}"
    assert sources[2] == f"live:voice-id:{MODEL}:low"
    assert used == {"s1"}


def test_session_picked_person_keeps_memories_until_another_voice_is_heard(store):
    from examples.shared.person_memory import load_person_context

    alice = memories.create_person(store, "Alice")
    _memory(store, alice, "Alice aime le thé.")

    async def run():
        h = _Harness(store, initial_person=await load_person_context(alice["id"]))
        assert h.session.current_speaker().trusted_person_id == alice["id"]
        assert await h.turn("Oui", tier="none") == "[speaker: Alice (uncertain)] Oui"
        assert h.pinned["content"] == "BASE|VOICE|Alice:- Alice aime le thé."
        await h.turn("Salut", provisional_id="unk-1", tier="unknown")
        assert h.pinned["content"] == "BASE|VOICE"

    asyncio.run(run())


def test_enroll_unknown_speaker(store):
    emb = voice_id.encode_vector(_vec(1))

    async def run():
        h = _Harness(store)
        assert "error" in await h.session._enroll("Bob")  # voice ID not active yet
        await h.turn("Bonjour", provisional_id="unk-1", tier="unknown")
        assert "full sentence" in (await h.session._enroll("Bob"))["error"]  # no final embedding yet
        assert "error" in await h.session._enroll("  ")
        await h.turn("Je m'appelle Bob", provisional_id="unk-1", tier="unknown", final=True, embedding=emb)

        assert await h.session._enroll(" bob ") == {"status": "enrolled", "name": "bob"}
        await h.drain()
        return h

    h = asyncio.run(run())
    bob = memories.find_person_by_name(store, "Bob")
    assert bob is not None
    assert h.session.current_speaker().trusted_person_id == bob["id"]
    assert memories.speakers_for(store, "s1")["turns"][2] == bob["id"]
    (frame,) = h.frames
    assert frame.data == {
        "type": "speaker-enrolled",
        "model": MODEL,
        "provisional_id": "unk-1",
        "person_id": bob["id"],
        "name": "bob",
        "centroid": frame.data["centroid"],
        "count": 1,
    }
    assert np.allclose(voice_id.decode_vector(frame.data["centroid"]), _vec(1), atol=1e-6)
    assert voice_id.gallery(store, MODEL)[0]["person_id"] == bob["id"]
    assert h.pinned["content"].startswith("BASE|VOICE|bob:")


def test_enroll_binds_to_existing_person_and_protects_confident_matches(store):
    alice = memories.create_person(store, "Alice")
    bob = memories.create_person(store, "Bob")

    async def run():
        h = _Harness(store)
        emb = voice_id.encode_vector(_vec(1))
        await h.turn("Bonjour", provisional_id="unk-1", tier="unknown", final=True, embedding=emb)
        assert (await h.session._enroll("ALICE"))["status"] == "enrolled"  # created in the review UI beforehand
        assert len(memories.list_people(store)) == 2

        # Confidently recognised as Alice: cannot be rebound to Bob, can be re-enrolled as Alice.
        emb2 = voice_id.encode_vector(_vec(2))
        await h.turn("Encore moi", person_id=alice["id"], tier="high", final=True, embedding=emb2)
        assert "already recognised as Alice" in (await h.session._enroll("Bob"))["error"]
        assert (await h.session._enroll("Alice"))["status"] == "enrolled"

        # Uncertain match as Alice who says they are Bob: only this turn's voice sample is used.
        emb3 = voice_id.encode_vector(_vec(3))
        await h.turn("Je suis Bob", person_id=alice["id"], tier="low", final=True, embedding=emb3)
        assert (await h.session._enroll("Bob"))["status"] == "enrolled"
        await h.drain()

    asyncio.run(run())
    found = voice_id.centroids(store, MODEL)
    assert found[bob["id"]]["count"] == 1
    assert np.allclose(found[bob["id"]]["centroid"], _vec(3), atol=1e-6)
    assert found[alice["id"]]["count"] == 2  # each sample is stored once


def test_voice_id_available(store, monkeypatch):
    assert live.voice_id_available()
    monkeypatch.setattr(memories, "live_store", lambda: None)
    assert not live.voice_id_available()

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(memories, "live_store", boom)
    assert not live.voice_id_available()


def test_speaker_turn_processor_tags_the_turn_before_the_llm(store):
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.tests.utils import run_test

    alice = memories.create_person(store, "Alice")

    async def run():
        h = _Harness(store)
        h.session.on_user_turn_started()
        await h.session.on_speaker_update(_payload(person_id=alice["id"], tier="high"))
        h.context.add_message({"role": "user", "content": "Bonjour"})
        processor = live.SpeakerTurnProcessor(h.session)
        down, _ = await run_test(
            processor,
            frames_to_send=[LLMContextFrame(h.context), LLMContextFrame(h.context)],
            expected_down_frames=[LLMContextFrame, LLMContextFrame],
        )
        await h.drain()
        return down[0].context.get_messages()[-1]["content"]

    # The second context frame (no open turn) leaves the message alone.
    assert asyncio.run(run()) == "[speaker: Alice] Bonjour"


# --------------------------------------------------------------------- face
def _face(person_id=None, link="doa", **overrides) -> dict:
    return {"model": FACE_MODEL, "person_id": person_id, "score": 0.6, "link": link, **overrides}


def _presence(*people) -> dict:
    return {
        "model": FACE_MODEL,
        "people": [
            {"person_id": pid, "provisional_id": prov, "score": 0.6, "tier": tier} for pid, prov, tier in people
        ],
    }


def test_gallery_serves_face_vectors_under_their_own_model_key(store):
    app = FastAPI()
    app.include_router(voice_id.create_voice_id_router())
    client = TestClient(app)
    alice = memories.create_person(store, "Alice")
    voice_id.add_embeddings(store, alice["id"], MODEL, [_vec(1)], source="t")
    voice_id.add_embeddings(store, alice["id"], FACE_MODEL, [np.ones(512), np.ones(512)], source="t")

    faces = client.get("/api/voice-id/gallery", params={"model": FACE_MODEL}).json()["people"]
    assert [(p["person_id"], p["count"]) for p in faces] == [(alice["id"], 2)]
    assert voice_id.decode_vector(faces[0]["centroid"]).size == 512
    voices = client.get("/api/voice-id/gallery", params={"model": MODEL}).json()["people"]
    assert voices[0]["count"] == 1 and voice_id.decode_vector(voices[0]["centroid"]).size == DIM


def test_parse_speaker_update_face_fields_and_verified_tier():
    emb = voice_id.encode_vector(_vec(5))
    verified = _update(person_id="p1", tier="verified", face=_face("p1"), final=True, face_embedding=emb)
    assert verified.tier == "verified"
    assert (verified.face.model, verified.face.person_id, verified.face.link) == (FACE_MODEL, "p1", "doa")
    assert np.allclose(verified.face_embedding, _vec(5), atol=1e-6)

    # ``verified`` without a linked face naming the same person is only a voice match.
    assert _update(person_id="p1", tier="verified").tier == "high"
    assert _update(person_id="p1", tier="verified", face=_face("p2")).tier == "high"
    assert _update(person_id="p1", tier="verified", face=_face(None)).tier == "high"
    assert _update(tier="verified", provisional_id="unk-1", face=_face(None)).tier == "unknown"

    # Malformed face evidence is dropped, and a face embedding needs both a face and a final update.
    assert _update(person_id="p1", tier="high", face={"model": FACE_MODEL, "link": "guess"}).face is None
    assert _update(person_id="p1", tier="high", face={"link": "doa"}).face is None
    assert _update(person_id="p1", tier="high", face="x", final=True, face_embedding=emb).face_embedding is None
    assert _update(person_id="p1", tier="high", face=_face("p1"), face_embedding=emb).face_embedding is None
    assert _update(person_id="p1", tier="high", face=_face(score=9)).face.score == 1.0


def test_parse_presence_update():
    assert parse_presence_update("nope") is None
    assert parse_presence_update({"model": "", "people": []}) is None
    assert parse_presence_update({"model": FACE_MODEL}) is None
    assert parse_presence_update(_presence(), now=1.0).people == ()

    presence = parse_presence_update(
        {
            "model": FACE_MODEL,
            "people": [
                {"person_id": "p1", "provisional_id": "unk-9", "score": 2, "tier": "high"},
                {"person_id": "p1", "tier": "low"},  # duplicate
                {"person_id": "p2", "tier": "verified"},  # not a face tier: unknown
                {"provisional_id": "unk-2", "tier": "high"},  # a match needs a person
                {"tier": "unknown"},
                "junk",
            ],
        },
        now=1.0,
    )
    assert [(p.person_id, p.provisional_id, p.tier) for p in presence.people] == [
        ("p1", None, "high"),
        (None, None, "unknown"),
        (None, "unk-2", "unknown"),
        (None, None, "unknown"),
    ]
    assert presence.people[0].score == 1.0
    crowd = parse_presence_update(_presence(*[(f"p{i}", None, "high") for i in range(40)]))
    assert len(crowd.people) == live.MAX_PRESENT_PEOPLE


def test_verified_speaker_is_trusted_and_falls_back_to_low():
    tracker = VoiceIdTracker()
    tracker.start_turn(now=1.0)
    tracker.on_update(_update(now=1.5, person_id="p1", tier="verified", face=_face("p1")))
    speaker = tracker.commit_turn(now=2.0)
    assert speaker.verified and speaker.trusted_person_id == "p1" and speaker.face_model == FACE_MODEL
    assert tracker.face_active and tracker.face_model == FACE_MODEL
    assert not Speaker(person_id="p1", tier="high").verified
    assert speaker_label(speaker, "Alice") == "Alice"

    tracker.start_turn(now=10.0)
    speaker = tracker.commit_turn(now=11.0)
    assert (speaker.person_id, speaker.tier, speaker.verified, speaker.is_guest) == ("p1", "low", False, True)


def test_tracker_buffers_face_embeddings_only_for_the_attributed_speaker():
    tracker = VoiceIdTracker()
    voice, face = voice_id.encode_vector(_vec(1)), voice_id.encode_vector(_vec(2))
    tracker.start_turn(now=1.0)
    # Voice says p1 but the linked face is recognised as p2: that face is not p1's.
    tracker.on_update(
        _update(now=1.5, person_id="p1", tier="low", final=True, embedding=voice, face=_face("p2"), face_embedding=face)
    )
    tracker.commit_turn(now=2.0)
    assert len(tracker.enrollment_sample()) == 1 and tracker.face_enrollment_sample() == []

    tracker.start_turn(now=10.0)
    tracker.on_update(
        _update(now=10.5, person_id="p1", tier="high", final=True, embedding=voice, face=_face(), face_embedding=face)
    )
    tracker.commit_turn(now=11.0)
    assert len(tracker.face_enrollment_sample()) == 1
    assert len(tracker.face_enrollment_sample(latest_turn_only=True)) == 1
    tracker.mark_enrolled("p1")
    assert tracker.face_enrollment_sample() == [] and tracker.enrollment_sample() == []


def test_session_verified_turn_loads_memories_and_records_its_source(store):
    alice = memories.create_person(store, "Alice")
    _memory(store, alice, "Alice aime le thé.")

    async def run():
        h = _Harness(store)
        tagged = await h.turn("Bonjour", person_id=alice["id"], tier="verified", face=_face(alice["id"]))
        assert tagged == "[speaker: Alice] Bonjour"
        assert h.pinned["content"] == "BASE|VOICE|FACE|Alice:- Alice aime le thé."
        assert h.session.current_speaker().verified
        # The face prompt block appears with the first face evidence, not before.
        h2 = _Harness(store)
        h2.turn_idx = 10
        await h2.turn("Bonjour", person_id=alice["id"], tier="high")
        assert h2.pinned["content"].startswith("BASE|VOICE|Alice")
        await h2.turn("Encore", person_id=alice["id"], tier="high", face=_face(None, link="single"))
        assert h2.pinned["content"].startswith("BASE|VOICE|FACE|Alice")

    asyncio.run(run())
    from sqlalchemy import select

    from monitoring import schema

    with store.engine.connect() as conn:
        sa = schema.speaker_assignments
        sources = dict(conn.execute(select(sa.c.turn_idx, sa.c.source)).all())
    assert sources[1] == f"live:voice-id:{MODEL}:verified"


def test_presence_is_named_in_the_speaker_tag_without_changing_the_speaker(store, monkeypatch):
    monkeypatch.setattr(live, "GREETING_WINDOW_SECS", 0.0)  # greetings are covered separately
    alice = memories.create_person(store, "Alice")
    bob = memories.create_person(store, "Bob")
    carol = memories.create_person(store, "Carol")
    _memory(store, bob, "Bob a un chat.")

    async def run():
        h = _Harness(store)
        assert await h.turn("Bonjour") == "Bonjour"  # inert before any voice or face message
        await h.session.on_presence_update({"model": FACE_MODEL})  # malformed: still inert
        assert not h.session.tracker.active

        await h.session.on_presence_update(
            _presence(
                (alice["id"], None, "high"),
                (bob["id"], None, "high"),
                (carol["id"], None, "low"),
                (None, "unk-1", "unknown"),
                (None, "unk-2", "unknown"),
                ("ghost", None, "high"),
            )
        )
        assert h.pinned["content"] == "BASE|VOICE|FACE"  # presence never loads memories
        assert h.session.current_speaker().is_guest and h.session.current_speaker().person_id is None

        tagged = await h.turn("Salut", person_id=alice["id"], tier="high")
        assert tagged == "[speaker: Alice; also in view: Bob, Carol (uncertain), 3 unknown guests] Salut"
        assert "Bob a un chat" not in h.pinned["content"]

        # An unknown speaker is not repeated among the people in view (shared provisional id).
        tagged = await h.turn("Hello", provisional_id="unk-1", tier="unknown")
        assert tagged.startswith("[speaker: unknown guest; also in view: Alice, Bob, Carol (uncertain), 2 unknown")

        await h.session.on_presence_update(_presence())
        assert await h.turn("Seul", person_id=alice["id"], tier="high") == "[speaker: Alice] Seul"

    asyncio.run(run())


def test_person_coming_into_view_is_greeted_once_and_only_while_idle(store, monkeypatch):
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        LLMRunFrame,
        UserStartedSpeakingFrame,
    )

    monkeypatch.setattr(live, "GREETING_QUIET_SECS", 0.0)
    monkeypatch.setattr(live, "GREETING_POLL_SECS", 0.01)
    alice = memories.create_person(store, "Alice")
    bob = memories.create_person(store, "Bob")
    carol = memories.create_person(store, "Carol")
    dave = memories.create_person(store, "Dave")

    async def run():
        h = _Harness(store)
        seen = [(alice["id"], None, "high"), (carol["id"], None, "low"), (None, "unk-1", "unknown")]
        await h.session.on_presence_update(_presence(*seen))
        await h.drain()
        assert [type(f) for f in h.frames] == [LLMRunFrame]
        assert h.context.get_messages()[-1] == {"role": "user", "content": "[presence: Alice came into view]"}
        assert h.pinned["content"] == "BASE|VOICE|FACE"  # greeted by name, without memories
        assert h.session.current_speaker().is_guest

        # Once per person per session, even after leaving and coming back.
        h.session.on_activity(BotStoppedSpeakingFrame())
        await h.session.on_presence_update(_presence())
        await h.session.on_presence_update(_presence(*seen))
        await h.drain()
        assert len(h.frames) == 1

        # Not while the bot speaks: Bob is greeted once it stopped ...
        h.session.on_activity(BotStartedSpeakingFrame())
        await h.session.on_presence_update(_presence((bob["id"], None, "high")))
        await asyncio.sleep(0.05)
        assert len(h.frames) == 1
        h.session.on_activity(BotStoppedSpeakingFrame())
        await h.drain()
        assert len(h.frames) == 2 and "Bob came into view" in h.context.get_messages()[-1]["content"]

        # ... and not at all when the conversation stays busy or the person left meanwhile.
        monkeypatch.setattr(live, "GREETING_WINDOW_SECS", 0.05)
        h.session.on_activity(UserStartedSpeakingFrame())
        await h.session.on_presence_update(_presence((dave["id"], None, "high")))
        await h.drain()
        assert len(h.frames) == 2

    asyncio.run(run())


def test_speaker_and_open_turn_are_not_greeted(store, monkeypatch):
    monkeypatch.setattr(live, "GREETING_QUIET_SECS", 0.0)
    monkeypatch.setattr(live, "GREETING_POLL_SECS", 0.01)
    monkeypatch.setattr(live, "GREETING_WINDOW_SECS", 0.05)
    alice = memories.create_person(store, "Alice")
    bob = memories.create_person(store, "Bob")

    async def run():
        h = _Harness(store)
        await h.turn("Bonjour", person_id=alice["id"], tier="high")
        from pipecat.frames.frames import BotStoppedSpeakingFrame

        h.session.on_activity(BotStoppedSpeakingFrame())
        await h.session.on_presence_update(_presence((alice["id"], None, "high")))  # already talking with her
        h.session.on_user_turn_started()  # a turn is open when Bob appears
        await h.session.on_presence_update(_presence((alice["id"], None, "high"), (bob["id"], None, "high")))
        await h.drain()
        assert h.frames == []

    asyncio.run(run())


def test_enroll_binds_face_with_voice_and_works_without_face(store):
    voice, face = voice_id.encode_vector(_vec(1)), voice_id.encode_vector(_vec(2))

    async def run():
        h = _Harness(store)
        await h.turn(
            "Je m'appelle Bob",
            provisional_id="unk-1",
            tier="unknown",
            final=True,
            embedding=voice,
            face=_face(None),
            face_embedding=face,
        )
        assert await h.session._enroll("Bob") == {"status": "enrolled", "name": "Bob"}
        # Voice only: no ``face`` object in the reply.
        await h.turn("Moi c'est Zoé", provisional_id="unk-2", tier="unknown", final=True, embedding=voice)
        assert (await h.session._enroll("Zoé"))["status"] == "enrolled"
        await h.drain()
        return h

    h = asyncio.run(run())
    bob = memories.find_person_by_name(store, "Bob")
    with_face, voice_only = (f.data for f in h.frames)
    assert with_face["face"] == {"model": FACE_MODEL, "centroid": with_face["face"]["centroid"], "count": 1}
    assert np.allclose(voice_id.decode_vector(with_face["face"]["centroid"]), _vec(2), atol=1e-6)
    assert (with_face["model"], with_face["count"], with_face["provisional_id"]) == (MODEL, 1, "unk-1")
    assert "face" not in voice_only
    assert [p["person_id"] for p in voice_id.gallery(store, FACE_MODEL)] == [bob["id"]]
    assert len(voice_id.gallery(store, MODEL)) == 2


def test_speaker_turn_processor_tracks_bot_speech(store):
    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.tests.utils import run_test

    async def run():
        h = _Harness(store)
        processor = live.SpeakerTurnProcessor(h.session)
        await run_test(
            processor,
            frames_to_send=[BotStartedSpeakingFrame()],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_up_frames=[BotStartedSpeakingFrame],
        )
        assert not h.session.conversation_idle(now=1e12)
        h.session.on_activity(BotStoppedSpeakingFrame())
        assert h.session.conversation_idle(now=1e12)

    asyncio.run(run())


# ----------------------------------------------------------- reconciliation
def _enroll(store, person, model, vectors, kind):
    voice_id.add_embeddings(store, person["id"], model, vectors, source=voice_id.ENROLL_SOURCES[kind], session_id="s1")


def test_identity_summary_labels_modality_and_falls_back_for_untagged_samples(store):
    alice = memories.create_person(store, "Alice")
    _enroll(store, alice, MODEL, [_vec(1), _vec(2)], "voice")
    _enroll(store, alice, FACE_MODEL, [_vec(3)], "face")
    voice_id.add_embeddings(store, alice["id"], "legacy@1", [_vec(4)], source="live:enroll")

    summary = {entry["model"]: entry for entry in voice_id.identity_summary(store, alice["id"])}

    assert {model: (e["modality"], e["count"]) for model, e in summary.items()} == {
        MODEL: ("voice", 2),
        FACE_MODEL: ("face", 1),
        "legacy@1": (None, 1),
    }
    assert summary[MODEL]["sessions"] == ["s1"]
    assert voice_id.identity_counts(store) == {alice["id"]: {"voice": 2, "face": 1, "other": 1}}


def test_duplicate_candidates_rank_similar_voices_and_faces_and_flag_same_names(store):
    alice = memories.create_person(store, "Alice")
    alice_again = memories.create_person(store, "Alice")
    alicia = memories.create_person(store, "Alicia")
    bob = memories.create_person(store, "Bob")
    near = voice_id.normalize(_vec(1) + 0.2 * _vec(9))
    _enroll(store, alice, MODEL, [_vec(1)], "voice")
    _enroll(store, alicia, MODEL, [near], "voice")
    _enroll(store, alice, FACE_MODEL, [_vec(5)], "face")
    _enroll(store, alicia, FACE_MODEL, [_vec(5)], "face")
    _enroll(store, bob, MODEL, [-_vec(1)], "voice")

    pairs = voice_id.duplicate_candidates(store)

    first = pairs[0]
    assert {p["id"] for p in first["people"]} == {alice["id"], alicia["id"]}
    assert {s["modality"] for s in first["scores"]} == {"voice", "face"}
    assert all(s["score"] > 0.9 for s in first["scores"])
    same_name = [p for p in pairs if p["same_name"]]
    assert [{x["id"] for x in p["people"]} for p in same_name] == [{alice["id"], alice_again["id"]}]
    assert all(bob["id"] not in {x["id"] for x in p["people"]} for p in pairs)
    assert all(
        bob["id"] in {x["id"] for x in p["people"]} for p in voice_id.duplicate_candidates(store, person_id=bob["id"])
    )


def test_merge_moves_everything_and_old_ids_keep_resolving(store):
    alice = memories.create_person(store, "Alice")
    duplicate = memories.create_person(store, "Alice B")
    carol = memories.create_person(store, "Carol")
    _enroll(store, duplicate, MODEL, [_vec(1)], "voice")
    _memory(store, duplicate, "Elle aime le thé.")
    memories.assign_speaker(store, "s1", duplicate["id"], source="live:voice-id:x", turn_idx=3)

    merged = memories.merge_people(store, duplicate["id"], alice["id"], annotator="rev")

    assert merged["moved"] == {"turns": 1, "embeddings": 1, "memories": 1}
    assert memories.get_person(store, duplicate["id"]) is None
    assert memories.speakers_for(store, "s1")["turns"] == {3: alice["id"]}
    assert [m["text"] for m in memories.active_memories(store, alice["id"])] == ["Elle aime le thé."]
    assert set(voice_id.centroids(store, MODEL)) == {alice["id"]}

    # A client that still has the old id attributes turns to the merged person.
    memories.assign_speaker(store, "s1", duplicate["id"], source="live:voice-id:x", turn_idx=4)
    assert memories.speakers_for(store, "s1")["turns"][4] == alice["id"]

    # Chained merges keep resolving to the final person.
    memories.merge_people(store, alice["id"], carol["id"], annotator="rev")
    assert memories.resolve_person_id(store, duplicate["id"]) == carol["id"]

    with pytest.raises(ValueError):
        memories.merge_people(store, carol["id"], carol["id"], annotator="rev")
    memories.update_person(store, carol["id"], archived=True)
    other = memories.create_person(store, "Dan")
    with pytest.raises(ValueError):
        memories.merge_people(store, other["id"], carol["id"], annotator="rev")
    with pytest.raises(KeyError):
        memories.merge_people(store, "nope", other["id"], annotator="rev")


def test_session_resolves_a_person_merged_while_the_client_kept_its_old_gallery(store):
    alice = memories.create_person(store, "Alice")
    duplicate = memories.create_person(store, "Alice B")
    _memory(store, duplicate, "Alice aime le thé.")
    memories.merge_people(store, duplicate["id"], alice["id"], annotator="rev")

    async def run():
        h = _Harness(store)
        assert await h.turn("Bonjour", person_id=duplicate["id"], tier="high") == "[speaker: Alice] Bonjour"
        assert "Alice aime le thé" in h.pinned["content"]
        assert h.session.current_speaker().trusted_person_id == alice["id"]

    asyncio.run(run())
    assert memories.speakers_for(store, "s1")["turns"] == {1: alice["id"]}


def test_review_api_identity_duplicates_merge_and_forget(store, tmp_path):
    from monitoring.api import create_review_router
    from monitoring.config import MonitoringConfig
    from monitoring.store import LocalArtifactStore

    settings = MonitoringConfig(
        enabled=True,
        data_dir=tmp_path,
        db_url=str(store.engine.url),
        record_audio_turns=False,
        record_audio_stereo=False,
        record_video="off",
        video_fps=1.0,
    )
    app = FastAPI()
    app.include_router(create_review_router(settings, store, LocalArtifactStore(settings.artifacts_dir), {}))
    client = TestClient(app)
    alice = memories.create_person(store, "Alice")
    alicia = memories.create_person(store, "Alicia")
    _enroll(store, alice, MODEL, [_vec(1)], "voice")
    _enroll(store, alicia, MODEL, [_vec(1)], "voice")
    _enroll(store, alicia, FACE_MODEL, [_vec(2)], "face")
    memories.assign_speaker(store, "s1", alicia["id"], source=f"live:voice-id:{MODEL}:verified", turn_idx=1)
    memories.assign_speaker(store, "s1", alicia["id"], source=f"live:voice-id:{MODEL}", turn_idx=2)

    people = {p["name"]: p for p in client.get("/api/review/people").json()["people"]}
    assert people["Alicia"]["identity_samples"] == {"voice": 1, "face": 1}
    assert people["Alicia"]["voice_id_turns"] == {"turns": 2, "verified": 1}
    identity = client.get(f"/api/review/people/{alicia['id']}/identity").json()
    assert {m["modality"] for m in identity["models"]} == {"voice", "face"}
    assert len(identity["duplicates"]) == 1
    assert len(client.get("/api/review/people/duplicates").json()["duplicates"]) == 1
    assert client.get("/api/review/people/nope/identity").status_code == 404

    url = f"/api/review/people/{alice['id']}/merge"
    assert client.post(url, json={"annotator": "", "source_id": alicia["id"]}).status_code == 422
    assert client.post(url, json={"annotator": "rev", "source_id": alice["id"]}).status_code == 422
    assert client.post(url, json={"annotator": "rev", "source_id": "nope"}).status_code == 404
    merged = client.post(url, json={"annotator": "rev", "source_id": alicia["id"]}).json()
    assert merged["moved"] == {"turns": 2, "embeddings": 2, "memories": 0}
    assert [p["name"] for p in client.get("/api/review/people").json()["people"]] == ["Alice"]

    forget = f"/api/review/people/{alice['id']}/forget-identity"
    assert client.post(forget, json={"annotator": "rev", "model": FACE_MODEL}).json() == {"deleted": 1}
    assert client.post(forget, json={"annotator": "rev"}).json() == {"deleted": 2}
    assert client.get("/api/review/people").json()["people"][0]["identity_samples"] == {}


def test_session_follows_a_merge_made_while_it_is_running(store):
    alice = memories.create_person(store, "Alice")
    duplicate = memories.create_person(store, "Alice B")
    _memory(store, alice, "Alice aime le thé.")

    async def run():
        h = _Harness(store)
        assert await h.turn("Bonjour", person_id=duplicate["id"], tier="high") == "[speaker: Alice B] Bonjour"
        memories.merge_people(store, duplicate["id"], alice["id"], annotator="rev")
        assert await h.turn("Re", person_id=duplicate["id"], tier="high") == "[speaker: Alice] Re"
        assert "Alice aime le thé" in h.pinned["content"]
        assert h.session.current_speaker().trusted_person_id == alice["id"]

    asyncio.run(run())

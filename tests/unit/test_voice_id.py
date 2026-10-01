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
    parse_speaker_update,
    speaker_label,
    tag_user_message,
    with_enroll_tool,
)
from monitoring import memories, voice_id
from monitoring.store import SessionStore

MODEL = "test-model@1"
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
    def _render(person, voice_id_active=False):
        content = "BASE"
        if voice_id_active:
            content += "|VOICE"
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

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from test_monitoring_memories import FakeLlm

from examples.shared import messaging_intents, messaging_tools
from examples.shared.intent_engine import learned as learned_module
from examples.shared.intent_engine.config import PACKAGED_SENTENCES_DIR
from examples.shared.intent_engine.learned import (
    MESSAGING_INTENTS,
    MESSAGING_SENTENCE_LISTS,
    LearnedIntent,
    load_learned_intents,
    merge_learned,
    sentence_problems,
)
from examples.shared.intent_engine.matcher import IntentMatcher, custom_sentence_data
from messaging.contacts import parse_contacts
from monitoring import intents, memories, schema
from monitoring.api import create_review_router
from monitoring.config import MonitoringConfig
from monitoring.jobs import JobContext, dream_messaging
from monitoring.jobs.dream import DreamJob
from monitoring.jobs.dream_messaging import MessagingPolicy, similar_message
from monitoring.jobs.runner import _DEFAULTS
from monitoring.store import LocalArtifactStore, SessionStore

CONTACTS = {
    "contacts": [
        {"name": "Bob Martin", "matrix_id": "@bob:matrix.org", "aliases": ["Bob"]},
        {"name": "Alice", "matrix_id": "@alice:matrix.org"},
    ]
}
BOB, ALICE = "@bob:matrix.org", "@alice:matrix.org"
PRIVATE_WORDS = ("bob", "martin", "alice", "retard", "demain", "late", "tomorrow")

# Per language: the session turns (user text, tool calls) and the dreamer's answer.
SESSIONS = {
    "fr": {
        "turns": {
            1: ("T'as reçu quelque chose de nouveau ?", [("check_messages", {})]),
            2: (
                "Balance à Bob que je serai en retard ce soir.",
                [("send_message", {"contact": "Bob", "text": "Je serai en retard ce soir."})],
            ),
            3: ("Carrément.", [("confirm_send_message", {})]),
            4: ("Qu'a raconté Alice ?", [("read_messages", {"contact": "Alice"})]),
            5: (
                "Renvoie-lui que d'accord pour demain",
                [("send_message", {"contact": "Alice", "text": "D'accord pour demain"})],
            ),
        },
        "answer": {
            "sentences": [
                {"intent": "MessagingCheck", "sentences": ["t'as reçu quelque chose [de nouveau]"], "turns": [1]},
                # The literal name becomes {contact}.
                {"intent": "MessagingSend", "sentences": ["balance à bob que {message}"], "turns": [2]},
                {"intent": "MessagingRead", "sentences": ["qu'a raconté alice"], "turns": [4]},
                {"intent": "MessagingReply", "sentences": ["renvoie-lui que {message}"], "turns": [5]},
                # Dialog answers and capture steps are never learned.
                {"intent": "MessagingYes", "sentences": ["carrément"], "turns": [3]},
                {"intent": "MessagingDictate", "sentences": ["balance {message}"], "turns": [2]},
                # {message} not at the end.
                {"intent": "MessagingSend", "sentences": ["{message} pour {contact}"], "turns": [2]},
                # The message copied into the sentence.
                {
                    "intent": "MessagingSend",
                    "sentences": ["balance à {contact} que je serai en retard ce soir"],
                    "turns": [2],
                },
                # A contact name {contact} does not cover ("Martin" of "Bob Martin").
                {"intent": "MessagingSend", "sentences": ["balance à martin que {message}"], "turns": [2]},
            ],
            "intents": [],
        },
        "learned": {
            "MessagingCheck": "t'as reçu quelque chose [de nouveau]",
            "MessagingSend": "balance à {contact} que {message}",
            "MessagingRead": "qu'a raconté {contact}",
            "MessagingReply": "renvoie-lui que {message}",
        },
        "evidence": {
            "MessagingSend": "Balance à {contact} que {message}",
            "MessagingRead": "Qu'a raconté {contact} ?",
            "MessagingReply": "Renvoie-lui que {message}",
        },
        "fresh": ("Balance à Bob que on se voit demain", "on se voit demain"),
    },
    "en": {
        "turns": {
            1: ("Got anything new for me?", [("check_messages", {})]),
            2: (
                "Shoot Bob a note saying I'll be late tonight.",
                [("send_message", {"contact": "Bob", "text": "I'll be late tonight."})],
            ),
            3: ("Totally.", [("confirm_send_message", {})]),
            4: ("What's new from Alice?", [("read_messages", {"contact": "Alice"})]),
            5: (
                "Write back saying see you tomorrow",
                [("send_message", {"contact": "Alice", "text": "See you tomorrow"})],
            ),
        },
        "answer": {
            "sentences": [
                {"intent": "MessagingCheck", "sentences": ["got anything new [for me]"], "turns": [1]},
                {"intent": "MessagingSend", "sentences": ["shoot bob a note saying {message}"], "turns": [2]},
                {"intent": "MessagingRead", "sentences": ["what's new from alice"], "turns": [4]},
                {"intent": "MessagingReply", "sentences": ["write back saying {message}"], "turns": [5]},
                {"intent": "MessagingYes", "sentences": ["totally"], "turns": [3]},
                {"intent": "MessagingNo", "sentences": ["nah"], "turns": [3]},
                {"intent": "MessagingSend", "sentences": ["saying {message} to {contact}"], "turns": [2]},
                {"intent": "MessagingSend", "sentences": ["shoot {contact} a note saying i'll be late"], "turns": [2]},
                {"intent": "MessagingSend", "sentences": ["shoot martin a note saying {message}"], "turns": [2]},
            ],
            "intents": [],
        },
        "learned": {
            "MessagingCheck": "got anything new [for me]",
            "MessagingSend": "shoot {contact} a note saying {message}",
            "MessagingRead": "what's new from {contact}",
            "MessagingReply": "write back saying {message}",
        },
        "evidence": {
            "MessagingSend": "Shoot {contact} a note saying {message}",
            "MessagingRead": "What's new from {contact}?",
            "MessagingReply": "Write back saying {message}",
        },
        "fresh": ("shoot Bob Martin a note saying the keys are under the mat", "the keys are under the mat"),
    },
}


@pytest.fixture
def contacts_file(tmp_path, monkeypatch):
    path = tmp_path / "contacts.yaml"
    path.write_text(json.dumps(CONTACTS), encoding="utf-8")
    monkeypatch.setenv("MATRIX_CONTACTS_FILE", str(path))
    return path


@pytest.fixture
def stores(tmp_path, monkeypatch, contacts_file):
    for name in ("HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN", "INTENT_ENGINE_SENTENCES_DIR"):
        monkeypatch.delenv(name, raising=False)
    store = SessionStore(f"sqlite:///{tmp_path / 'db.sqlite'}")
    store.create_schema()
    return store, LocalArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def llm():
    fake = FakeLlm({"sentences": [], "intents": []})
    yield fake
    fake.close()


def _session(store, sid, base_url, language, started=1000.0):
    store.create_session(
        {
            "id": sid,
            "example": "multilingual-assistant",
            "started_at": started,
            "ended_at": started + 60,
            "config": {
                "language": {"fr": "fr-FR", "en": "en-US"}[language],
                "client_tools": [],
                "intent_engine": {"enabled": True, "home_assistant": False},
                "llm": {"model": "llm", "base_url": base_url},
            },
            "artifact_prefix": f"sessions/{sid}/",
        }
    )
    rows = []
    for idx, (text, calls) in SESSIONS[language]["turns"].items():
        rows.append(("turns", {"session_id": sid, "idx": idx, "user_text": text, "bot_text": "OK."}))
        rows.append(
            (
                "events",
                {
                    "session_id": sid,
                    "turn_idx": idx,
                    "ts": started + idx,
                    "kind": "intent_engine",
                    "data": {"handled_by": "llm", "reason": "no_match"},
                },
            )
        )
        function_calls = [
            {"name": name, "tool_call_id": f"c{idx}{i}", "arguments": arguments}
            for i, (name, arguments) in enumerate(calls)
        ]
        rows.append(
            (
                "llm_calls",
                {"session_id": sid, "turn_idx": idx, "started_at": started + idx, "function_calls": function_calls},
            )
        )
    store.write_batch(rows)


def _run(store, artifacts, sid):
    store.reset_job("dream", sid)
    job = next(j for j in store.jobs() if j["kind"] == "dream" and j["target"] == sid)
    ctx = JobContext(
        job=job, store=store, artifacts=artifacts, config=dict(_DEFAULTS), is_preempted=lambda: False, progress={}
    )
    DreamJob().run(ctx)
    return ctx.progress


def _dream(stores, llm, language):
    store, artifacts = stores
    _session(store, "s1", llm.base_url, language)
    llm.answer = SESSIONS[language]["answer"]
    return _run(store, artifacts, "s1")


def _contacts():
    return parse_contacts(CONTACTS)


# ---------------------------------------------------------------- constants
def test_messaging_names_match_the_dialog_and_tools():
    assert MESSAGING_INTENTS == messaging_intents.MESSAGING_INTENTS
    assert set(MESSAGING_SENTENCE_LISTS) == {
        messaging_intents.CHECK_INTENT,
        messaging_intents.READ_INTENT,
        messaging_intents.SEND_INTENT,
        messaging_intents.REPLY_INTENT,
    }
    assert set(messaging_tools.MESSAGING_TOOL_NAMES) == dream_messaging.MESSAGING_TOOLS
    assert MESSAGING_SENTENCE_LISTS["MessagingSend"] == (
        {messaging_intents.CONTACT_SLOT},
        {messaging_intents.MESSAGE_SLOT},
    )


# ------------------------------------------------------- sentence problems
def test_message_wildcard_only_trailing():
    data = {"lists": {"message": {"wildcard": True}}}
    lists = {"extra_lists": {"contact"}, "trailing_wildcards": {"message"}}
    assert sentence_problems("dis à {contact} que {message}", data, **lists) == []
    assert sentence_problems("(dis|écris) {message}", data, **lists) == []
    for template in (
        "{message} pour {contact}",
        "{message}",
        "[que ]{message}",
        "dis {message} et {message}",
        "dis à {contact} [que {message}]",
    ):
        assert sentence_problems(template, data, **lists) == [
            "wildcard list {message} must end the sentence, once, after required words"
        ], template
    # Without the exemption a wildcard stays refused.
    assert sentence_problems("dis {message}", data) == ["wildcard list {message}"]


@pytest.mark.parametrize("language", ["fr", "en"])
def test_merge_learned_takes_request_intents_only(language):
    files = custom_sentence_data(PACKAGED_SENTENCES_DIR, language)
    rows = [
        LearnedIntent(id=1, kind="sentence", intent="MessagingSend", sentences=("zz {contact} zz {message}",)),
        LearnedIntent(id=2, kind="sentence", intent="MessagingYes", sentences=("zz",)),
        LearnedIntent(id=3, kind="sentence", intent="MessagingDictate", sentences=("zz",)),
        LearnedIntent(id=4, kind="sentence", intent="MessagingSend", sentences=("{message} zz {contact}",)),
        LearnedIntent(id=5, kind="sentence", intent="MessagingCheck", sentences=("zz {contact}",)),
        LearnedIntent(id=6, kind="sentence", intent="MessagingRead", sentences=("zz {message}",)),
        LearnedIntent(id=7, kind="sentence", intent="MessagingReply", sentences=("zz {message}",)),
    ]
    _, used = merge_learned(files, rows, add_intents=True)
    assert used == [1, 7]


# -------------------------------------------------------------- the policy
def test_policy_substitutes_names_and_finds_leftovers():
    policy = MessagingPolicy(_contacts(), [{"tool_calls": [{"name": "send_message", "arguments": {"contact": "Zoé"}}]}])
    assert (
        policy.substitute("écris à bob martin et à BOB que {message}")
        == "écris à {contact} et à {contact} que {message}"
    )
    assert policy.substitute("préviens zoe que {message}") == "préviens {contact} que {message}"
    assert policy.name_problems("préviens martin que {message}") == ["it names a contact; write {contact}"]
    assert policy.name_problems("préviens {contact} que {message}") == []
    assert policy.message_problems("dis à {contact} que je serai en retard", ["Je serai en retard ce soir"])
    assert policy.message_problems("dis à {contact} que {message}", ["Je serai en retard ce soir"]) == []
    assert policy.message_problems("dis ok à {contact}", ["ok"]) == []  # too short to tell


def test_policy_slot_extraction():
    policy = MessagingPolicy(_contacts())
    send = [{"name": "send_message", "arguments": {"contact": "Bob", "text": "J'arrive dans 5 minutes."}}]
    assert policy.check_turn("MessagingSend", {"contact": BOB, "message": "j'arrive dans 5 minutes"}, send) is None
    assert policy.check_turn("MessagingSend", {"contact": ALICE, "message": "j'arrive dans 5 minutes"}, send)
    assert policy.check_turn("MessagingSend", {"contact": BOB}, send)  # the message is not captured
    assert policy.check_turn("MessagingReply", {"message": "j'arrive dans cinq minutes"}, send) is None
    assert policy.check_turn("MessagingReply", {"message": "quelque chose d'autre"}, send)
    read = [{"name": "read_messages", "arguments": {"contact": "Bob Martin"}}]
    assert policy.check_turn("MessagingRead", {"contact": BOB}, read) is None
    assert policy.check_turn("MessagingRead", {}, read)
    assert policy.check_turn("MessagingRead", {}, [{"name": "read_messages", "arguments": {}}]) is None
    assert policy.check_turn("MessagingCheck", {}, [{"name": "check_messages", "arguments": {}}]) is None
    # A confirmation alone teaches nothing.
    assert policy.check_turn("MessagingSend", {"contact": BOB, "message": "x"}, [{"name": "confirm_send_message"}])
    assert similar_message("i ll be late", "I'll be late.")
    assert not similar_message("", "I'll be late.")


def test_policy_redacts_evidence():
    policy = MessagingPolicy(_contacts())
    calls = [{"name": "send_message", "arguments": {"contact": "Bob", "text": "Je serai en retard."}}]
    text = "Balance à Bob que je serai en retard, désolé !"
    slots = {"contact": BOB, "message": "je serai en retard désolé"}
    assert policy.redact_text(text, slots, calls) == "Balance à {contact} que {message}"
    assert policy.redact_calls(calls) == [
        {"name": "send_message", "arguments": {"contact": "{contact}", "text": "{message}"}}
    ]
    # A message that cannot be located is not stored at all.
    assert policy.redact_text("Balance à Bob un truc", {"contact": BOB, "message": "autre chose"}, []) is None


# ----------------------------------------------------------- the dream job
@pytest.mark.parametrize("language", ["fr", "en"])
def test_dream_learns_messaging_sentences(stores, llm, language):
    store, _ = stores
    progress = _dream(stores, llm, language)
    expected = SESSIONS[language]

    assert progress["intents"]["created"] == 4, progress
    assert progress["intents"]["dropped"] == 5
    prompt = llm.requests[-1]["messages"][1]["content"]
    assert "MessagingSend: runs the messaging dialog" in prompt
    assert "- {contact}: a contact's name" in prompt
    assert "MessagingYes" not in prompt and "MessagingDictate" not in prompt
    assert "{contact} for the contact and never a name" in llm.requests[-1]["messages"][0]["content"]

    listed = intents.list_suggestions(store)["suggestions"]
    learned = {s["intent"]: s["sentences"] for s in listed}
    assert learned == {intent: [sentence] for intent, sentence in expected["learned"].items()}
    for suggestion in listed:
        assert suggestion["validated"] is True and suggestion["slots"] is None
        for sentence in suggestion["sentences"]:
            assert not any(word in sentence.casefold() for word in PRIVATE_WORDS), sentence
    evidence = {s["intent"]: s["evidence"][0] for s in listed}
    for intent, text in expected["evidence"].items():
        assert evidence[intent]["text"] == text
        assert evidence[intent]["audio"] == []
    assert evidence["MessagingSend"]["tool_calls"] == [
        {"name": "send_message", "arguments": {"contact": "{contact}", "text": "{message}"}}
    ]

    # Nothing private is stored with the suggestions or their evidence.
    stored = json.dumps(
        [
            *store.query(select(schema.intent_suggestions)),
            *store.query(select(schema.intent_evidence)),
        ],
        default=str,
    ).casefold()
    assert not any(word in stored for word in PRIVATE_WORDS)


def test_dream_skips_messaging_without_contacts(stores, llm, monkeypatch, tmp_path):
    monkeypatch.setenv("MATRIX_CONTACTS_FILE", str(tmp_path / "missing.yaml"))
    progress = _dream(stores, llm, "fr")
    assert progress["intents"]["created"] == 0
    assert any("messaging contacts unavailable" in note for note in progress["intents"]["notes"])
    assert "MessagingSend" not in llm.requests[-1]["messages"][1]["content"]


@pytest.mark.parametrize("language", ["fr", "en"])
def test_approved_messaging_sentence_matches_in_a_new_session(stores, llm, monkeypatch, language):
    store, _ = stores
    _dream(stores, llm, language)
    for row in intents.list_suggestions(store)["suggestions"]:
        intents.review_suggestion(store, row["id"], "approve", reviewer="ann")

    monkeypatch.setattr(memories, "live_store", lambda: store)
    learned = load_learned_intents({"fr": "fr-FR", "en": "en-US"}[language])
    assert {item.intent for item in learned} == set(MESSAGING_SENTENCE_LISTS)
    matcher = IntentMatcher(
        language,
        allowed_intents=(),
        sentences_dir=PACKAGED_SENTENCES_DIR,
        load_stock=False,
        learned=learned,
        keep_custom_intent=lambda name: name in MESSAGING_INTENTS,
    )
    contacts = {"contact": messaging_intents.contact_slot_list(_contacts())}
    said, message = SESSIONS[language]["fresh"]
    match = matcher.match(said, extra_slot_lists=contacts)
    send = next(item for item in learned if item.intent == "MessagingSend")
    assert match.name == "MessagingSend" and match.learned_id == send.id
    assert match.slots == {"contact": BOB, "message": message}


def test_learned_messaging_sentences_load_through_learned_module(stores, llm, monkeypatch):
    # The loader warns and skips a learned dialog answer even if one reached the store.
    store, _ = stores
    monkeypatch.setattr(memories, "live_store", lambda: store)
    intents.add_suggestion(
        store,
        {
            "kind": "sentence",
            "language": "fr",
            "intent": "MessagingYes",
            "sentences": ["carrément"],
            "status": "active",
            "source": "human:ann",
        },
        [],
    )
    learned = learned_module.load_learned_intents("fr-FR")
    matcher = IntentMatcher(
        "fr-FR", allowed_intents=(), sentences_dir=PACKAGED_SENTENCES_DIR, load_stock=False, learned=learned
    )
    assert matcher.match("carrément", dialog="messaging") is None


@pytest.fixture
def api(tmp_path, llm, stores):
    store, artifacts = stores
    settings = MonitoringConfig(
        enabled=True,
        data_dir=tmp_path,
        db_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        record_audio_turns=True,
        record_audio_stereo=True,
        record_video="off",
        video_fps=1.0,
    )
    _dream(stores, llm, "fr")
    app = FastAPI()
    app.include_router(create_review_router(settings, store, artifacts, {**_DEFAULTS, "reasr": {}}))
    return TestClient(app), store


def test_review_api_shows_sentence_shapes_only(api):
    client, _ = api
    response = client.get("/api/review/intents")
    assert response.status_code == 200 and response.json()["total"] == 4
    assert not any(word in response.text.casefold() for word in PRIVATE_WORDS)

    send = next(s for s in response.json()["suggestions"] if s["intent"] == "MessagingSend")
    bad = client.post(
        f"/api/review/intents/{send['id']}/review",
        json={"annotator": "ann", "action": "correct", "sentences": ["préviens bob que {message}"]},
    )
    assert bad.status_code == 422 and "names a contact" in bad.text
    bad = client.post(
        f"/api/review/intents/{send['id']}/review",
        json={"annotator": "ann", "action": "correct", "sentences": ["{message} pour {contact}"]},
    )
    assert bad.status_code == 422 and "must end the sentence" in bad.text
    corrected = client.post(
        f"/api/review/intents/{send['id']}/review",
        json={"annotator": "ann", "action": "correct", "sentences": ["(balance|envoie) à {contact} que {message}"]},
    )
    assert corrected.status_code == 200 and corrected.json()["status"] == "active"
    assert not any(word in corrected.text.casefold() for word in PRIVATE_WORDS)

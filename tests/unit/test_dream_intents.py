# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import copy

import home_assistant_intents
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hassil import Intents, recognize_best
from test_monitoring_memories import FakeLlm

from examples.shared.intent_engine.config import PACKAGED_SENTENCES_DIR
from examples.shared.intent_engine.learned import LearnedIntent, merge_learned, sentence_problems, stock_blocks
from examples.shared.intent_engine.matcher import IntentMatcher, build_slot_lists, custom_sentence_data
from examples.shared.intent_engine.processor import build_intent_engine
from monitoring import intents
from monitoring.api import create_review_router
from monitoring.config import MonitoringConfig
from monitoring.jobs import JobContext
from monitoring.jobs.dream import DreamJob
from monitoring.jobs.dream_intents import parse_proposals
from monitoring.jobs.runner import _DEFAULTS
from monitoring.store import LocalArtifactStore, SessionStore

CLIENT_TOOLS = ["move_head", "dance", "stop_dance", "play_emotion", "volume_control", "head_tracking"]


@pytest.fixture
def stores(tmp_path, monkeypatch):
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


def _session(store, sid, base_url, *, engine=True, started=1000.0):
    store.create_session(
        {
            "id": sid,
            "example": "multilingual-assistant",
            "started_at": started,
            "ended_at": started + 60,
            "config": {
                "language": "fr-FR",
                "client_tools": CLIENT_TOOLS,
                "intent_engine": {"enabled": engine, "home_assistant": False, "wake_words": ["Reachy"]},
                "llm": {"model": "llm", "base_url": base_url},
            },
            "artifact_prefix": f"sessions/{sid}/",
        }
    )
    turns = {
        1: ("Reachy, tourne-toi vers la gauche", "move_head", {"direction": "left"}),
        2: ("quel temps fait-il", None, None),
        3: ("suis-moi du regard", "head_tracking", {"enabled": True}),
    }
    rows = [("turns", {"session_id": sid, "idx": 0, "bot_text": "Bonjour !"})]
    for idx, (text, tool, arguments) in turns.items():
        rows.append(("turns", {"session_id": sid, "idx": idx, "user_text": text, "bot_text": "D'accord."}))
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
        calls = [{"name": tool, "tool_call_id": f"c{idx}", "arguments": arguments}] if tool else []
        rows.append(
            ("llm_calls", {"session_id": sid, "turn_idx": idx, "started_at": started + idx, "function_calls": calls})
        )
    rows.append(("turns", {"session_id": sid, "idx": 4, "user_text": "danse", "bot_text": "C'est parti"}))
    rows.append(
        (
            "events",
            {
                "session_id": sid,
                "turn_idx": 4,
                "ts": started + 4,
                "kind": "intent_engine",
                "data": {"handled_by": "intent", "intent": "ReachyDance"},
            },
        )
    )
    store.write_batch(rows)


def _run(store, artifacts, sid, *, config=None, progress=None):
    store.reset_job("dream", sid)
    job = next(j for j in store.jobs() if j["kind"] == "dream" and j["target"] == sid)
    ctx = JobContext(
        job=job,
        store=store,
        artifacts=artifacts,
        config=config or dict(_DEFAULTS),
        is_preempted=lambda: False,
        progress=progress or {},
    )
    DreamJob().run(ctx)
    return ctx.progress


ANSWER = {
    "sentences": [
        {
            "intent": "ReachyMoveHead",
            "sentences": ["tourne-toi vers [la ]{direction}"],
            "slots": {},
            "turns": [1],
            "confidence": 0.95,
        },
        # Unknown list: dropped.
        {"intent": "ReachyMoveHead", "sentences": ["pivote {cote}"], "turns": [1], "confidence": 0.9},
        # Turn 4 was answered by the engine: not evidence.
        {"intent": "ReachyDance", "sentences": ["danse"], "turns": [4], "confidence": 0.9},
        # Matches the person's words, but the tool call was "left", not "right".
        {
            "intent": "ReachyMoveHead",
            "sentences": ["tourne-toi vers la gauche"],
            "slots": {"direction": "right"},
            "turns": [1],
            "confidence": 0.9,
        },
    ],
    "intents": [
        {
            "name": "ReachyFollowMe",
            "tool": "head_tracking",
            "arguments": {"enabled": True},
            "sentences": ["suis-moi [du regard]"],
            "response": "Je te suis",
            "turns": [3],
            "confidence": 0.9,
        },
        # Not a client tool of the session.
        {
            "name": "ReachyWeather",
            "tool": "get_weather",
            "arguments": {},
            "sentences": ["quel temps fait-il"],
            "response": "Voilà",
            "turns": [2],
        },
    ],
}


def test_parse_proposals_tolerates_think_tags():
    sentences, new = parse_proposals('<think>{"x": 1}</think>Sure: {"sentences": [{"intent": "A"}, 3]}')
    assert sentences == [{"intent": "A"}] and new == []
    with pytest.raises(ValueError):
        parse_proposals("nothing")


def test_sentence_problems():
    data = {"lists": {"direction": {}, "anything": {"wildcard": True}}, "expansion_rules": {"montre": "(montre)"}}
    assert sentence_problems("<montre> {direction}", data) == []
    assert sentence_problems("(oops", data)[0].startswith("does not parse")
    assert sentence_problems("<other> {nope} {anything}", data) == [
        "unknown list {nope}",
        "unknown rule <other>",
        "wildcard list {anything}",
    ]
    assert sentence_problems("allume {name}", data, extra_lists={"name"}) == []


def test_dream_learns_checked_sentences_and_intents(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    llm.answer = ANSWER
    progress = _run(store, artifacts, "s1")

    # No person attributed: memories are skipped, intents still learned.
    assert progress["skipped"] == "no speaker"
    assert progress["intents_written"] is True
    assert progress["intents"]["created"] == 2
    assert progress["intents"]["dropped"] == 4
    prompt = llm.requests[-1]["messages"][1]["content"]
    assert "ReachyMoveHead: runs client tool move_head" in prompt
    assert "[turn 1] User (missed): Reachy, tourne-toi vers la gauche" in prompt
    assert "[turn 4] User (answered by the engine: ReachyDance): danse" in prompt
    assert 'Tool call: move_head({"direction": "left"})' in prompt

    listed = intents.list_suggestions(store)["suggestions"]
    by_kind = {s["kind"]: s for s in listed}
    sentence = by_kind["sentence"]
    assert sentence["intent"] == "ReachyMoveHead" and sentence["status"] == "proposed"
    assert sentence["validated"] is True and sentence["occurrences"] == 1
    assert sentence["evidence"][0]["text"] == "Reachy, tourne-toi vers la gauche"
    new = by_kind["intent"]
    assert new["tool"] == {"tool": "head_tracking", "arguments": {"enabled": True}}
    assert new["response"] == "Je te suis" and new["validated"] is True


def test_dream_merges_recurrences_skips_rejected_and_reruns(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    _session(store, "s2", llm.base_url, started=2000.0)
    llm.answer = ANSWER
    _run(store, artifacts, "s1")
    progress = _run(store, artifacts, "s2")
    assert progress["intents"]["merged"] == 2 and progress["intents"]["created"] == 0
    listed = intents.list_suggestions(store)["suggestions"]
    assert [s["occurrences"] for s in listed] == [2, 2] and listed[0]["sessions"] == 2

    # A re-run replaces the session's evidence instead of adding to it.
    _run(store, artifacts, "s2")
    assert [s["occurrences"] for s in intents.list_suggestions(store)["suggestions"]] == [2, 2]

    new = next(s for s in listed if s["kind"] == "intent")
    intents.review_suggestion(store, new["id"], "reject", reviewer="ann")
    progress = _run(store, artifacts, "s1")
    assert progress["intents"]["rejected_before"] == 1
    assert intents.list_suggestions(store, status="rejected")["total"] == 1


def test_dream_skips_sessions_without_intent_engine(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url, engine=False)
    progress = _run(store, artifacts, "s1")
    assert progress["intents_skipped"].startswith("intent engine off")
    assert llm.requests == []  # neither memories nor intents needed the LLM


def test_auto_use_threshold_activates_validated_sentences(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    llm.answer = ANSWER
    store.reset_job("dream", "s1")
    job = next(j for j in store.jobs() if j["kind"] == "dream")
    config = {**_DEFAULTS, "dream": {"intents": {"auto_use_threshold": 0.9}}}
    ctx = JobContext(job=job, store=store, artifacts=artifacts, config=config, is_preempted=lambda: False)
    DreamJob().run(ctx)
    assert ctx.progress["intents"]["active"] == 1
    active = intents.active_suggestions(store, "fr-FR")
    assert [(s["kind"], s["intent"]) for s in active] == [("sentence", "ReachyMoveHead")]  # new intents wait


def test_live_matcher_uses_approved_suggestions(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    llm.answer = ANSWER
    _run(store, artifacts, "s1")
    for row in intents.list_suggestions(store)["suggestions"]:
        intents.review_suggestion(store, row["id"], "approve", reviewer="ann")
    learned = [LearnedIntent.from_row(r) for r in intents.active_suggestions(store, "fr")]

    matcher = IntentMatcher(
        "fr-FR", allowed_intents=(), sentences_dir=PACKAGED_SENTENCES_DIR, load_stock=False, learned=learned
    )
    match = matcher.match("tourne-toi vers la droite")
    assert match.name == "ReachyMoveHead" and match.slots == {"direction": "right"}
    assert match.learned_id == next(i.id for i in learned if i.kind == "sentence")
    assert matcher.match("regarde à gauche").learned_id is None  # packaged sentence

    engine = build_intent_engine(
        _engine_config(),
        language="fr-FR",
        context=_Context(),
        llm=None,
        client_tool_names=tuple(CLIENT_TOOLS),
        learned=learned,
    )
    assert "ReachyFollowMe" in engine._matcher.custom_intent_names
    assert engine._tool_map.tool("ReachyFollowMe") == "head_tracking"
    match = engine._matcher.match("suis-moi")
    assert match.name == "ReachyFollowMe"
    assert engine._matcher.response_template(match) == "Je te suis"

    # Without the tool declared by the client, the learned intent is inactive.
    engine = build_intent_engine(
        _engine_config(),
        language="fr-FR",
        context=_Context(),
        llm=None,
        client_tool_names=("move_head",),
        learned=learned,
    )
    assert "ReachyFollowMe" not in engine._matcher.custom_intent_names


def test_merge_learned_ignores_broken_suggestions():
    data = {"language": "fr", "intents": {"A": {"data": [{"sentences": ["a"]}]}}, "lists": {}}
    broken = LearnedIntent(id=7, kind="sentence", intent="A", sentences=("b {missing}",))
    other = LearnedIntent(id=8, kind="sentence", intent="Unknown", sentences=("c",))
    merged, used = merge_learned(data, [broken, other], add_intents=True)
    assert used == [] and merged["intents"]["A"]["data"] == [{"sentences": ["a"]}]


def _engine_config():
    from examples.shared.intent_engine.config import IntentEngineConfig

    return IntentEngineConfig(enabled=True)


class _Context:
    def get_messages(self):
        return []


@pytest.fixture
def api(tmp_path, llm, monkeypatch):
    for name in ("HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN", "INTENT_ENGINE_SENTENCES_DIR"):
        monkeypatch.delenv(name, raising=False)
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
    _session(store, "s1", llm.base_url)
    llm.answer = ANSWER
    _run(store, artifacts, "s1")
    app = FastAPI()
    app.include_router(create_review_router(settings, store, artifacts, {**_DEFAULTS, "reasr": {}}))
    return TestClient(app), store


def test_api_review_intents(api):
    client, store = api
    activities = {a["id"]: a for a in client.get("/api/review/activities").json()["activities"]}
    assert activities["intents"]["open"] == 2

    listed = client.get("/api/review/intents").json()
    assert listed["total"] == 2
    sentence = next(s for s in listed["suggestions"] if s["kind"] == "sentence")
    new = next(s for s in listed["suggestions"] if s["kind"] == "intent")

    bad = client.post(
        f"/api/review/intents/{sentence['id']}/review",
        json={"annotator": "ann", "action": "correct", "sentences": ["tourne {nope}"]},
    )
    assert bad.status_code == 422 and "unknown list" in bad.text

    corrected = client.post(
        f"/api/review/intents/{sentence['id']}/review",
        json={"annotator": "ann", "action": "correct", "sentences": ["(tourne|pivote)-toi vers [la ]{direction}"]},
    ).json()
    assert corrected["status"] == "active" and corrected["source"] == "human:ann"
    assert corrected["occurrences"] == 1  # evidence carried over

    clash = client.post(
        f"/api/review/intents/{new['id']}/review",
        json={"annotator": "ann", "action": "correct", "intent": "ReachyDance"},
    )
    assert clash.status_code == 422
    approved = client.post(
        f"/api/review/intents/{new['id']}/review", json={"annotator": "ann", "action": "approve"}
    ).json()
    assert approved["status"] == "active"

    exported = client.get("/api/review/intents/export", params={"language": "fr-FR"})
    assert exported.status_code == 200
    assert "ReachyFollowMe" in exported.text and "head_tracking" in exported.text
    assert client.get("/api/review/activities").json()["activities"][-1]["open"] == 0


def test_in_use_suggestion_keeps_its_sentences(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    # Auto-used above the threshold: active, not reviewed.
    active_id = intents.add_suggestion(store, _row("ReachyMoveHead", "pivote vers [la ]{direction}", "dream:llm"), [])

    # The session proposes the in-use sentence plus a new one.
    answer = copy.deepcopy(ANSWER)
    answer["sentences"][0]["sentences"].insert(0, "pivote vers [la ]{direction}")
    llm.answer = answer
    progress = _run(store, artifacts, "s1", config={**_DEFAULTS, "dream": {"intents": {"auto_use_threshold": 0.9}}})
    assert progress["intents"]["merged"] == 1
    assert intents.get_suggestion(store, active_id)["sentences"] == ["pivote vers [la ]{direction}"]
    extra = next(r for r in intents.list_suggestions(store)["suggestions"] if r["kind"] == "sentence")
    assert extra["sentences"] == ["tourne-toi vers [la ]{direction}"]
    assert extra["status"] == "proposed" and extra["validated"] is False
    assert [r["id"] for r in intents.active_suggestions(store, "fr")] == [active_id]


def _row(intent, sentence, source="human:ann"):
    return {
        "kind": "sentence",
        "language": "fr",
        "intent": intent,
        "sentences": [sentence],
        "status": "active",
        "source": source,
    }


def test_rerun_keeps_suggestions_when_the_llm_answer_is_unusable(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    llm.answer = ANSWER
    _run(store, artifacts, "s1")
    before = intents.list_suggestions(store)["total"]

    llm.answer = "sorry, I cannot help"
    with pytest.raises(ValueError):
        _run(store, artifacts, "s1", progress={"written": True})
    assert intents.list_suggestions(store)["total"] == before == 2

    # A re-run that finds nothing to learn retracts the earlier proposals.
    store.write_batch(
        [
            (
                "events",
                {"session_id": "s1", "turn_idx": idx, "ts": 1100.0 + idx, "kind": "intent_engine", "data": data},
            )
            for idx, data in ((1, {"handled_by": "intent", "intent": "ReachyMoveHead"}),)
            + tuple((i, {"handled_by": "llm", "reason": "blocked"}) for i in (2, 3))
        ]
    )
    progress = _run(store, artifacts, "s1", progress={"written": True})
    assert progress["intents_skipped"] == "no missed turn"
    assert intents.list_suggestions(store)["total"] == 0


def test_new_intent_name_must_not_clash_with_an_inactive_one(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    answer = copy.deepcopy(ANSWER)
    # ReachyGoToSleep is a packaged intent whose tool this session does not declare.
    answer["intents"][0]["name"] = "ReachyGoToSleep"
    llm.answer = answer
    progress = _run(store, artifacts, "s1")
    assert progress["intents"]["created"] == 1
    assert [r["kind"] for r in intents.list_suggestions(store)["suggestions"]] == ["sentence"]


def test_live_uses_and_order(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    llm.answer = ANSWER
    _run(store, artifacts, "s1")
    ids = [r["id"] for r in intents.list_suggestions(store)["suggestions"]]
    store.write_batch(
        [
            (
                "events",
                {
                    "session_id": "s1",
                    "turn_idx": 9,
                    "ts": 1200.0 + n,
                    "kind": "intent_engine",
                    "data": {"handled_by": handled, "intent": "X", "learned": ids[1]},
                },
            )
            for n, handled in enumerate(("intent", "intent", "llm"))
        ]
    )
    assert intents.live_uses(store, ids) == {ids[1]: 2}
    page = intents.list_suggestions(store, limit=1, offset=1)
    assert page["total"] == 2 and [r["id"] for r in page["suggestions"]] == [ids[1]]


def test_stock_sentences_keep_the_shape_of_the_stock_blocks():
    data = home_assistant_intents.get_intents("fr")
    body = data["intents"]["HassTurnOn"]
    by_name = LearnedIntent(id=5, kind="sentence", intent="HassTurnOn", sentences=("mets en marche [<le>]{name}",))
    blocks, misfits = stock_blocks(by_name, body, data)
    assert not misfits and blocks
    assert all(b.get("requires_context", {}).get("domain") for b in blocks)
    assert all(b["metadata"]["learned_id"] == 5 for b in blocks)

    # No domain: "turn on the living room" would turn on everything in it.
    by_area = LearnedIntent(id=6, kind="sentence", intent="HassTurnOn", sentences=("mets en marche {area}",))
    assert stock_blocks(by_area, body, data)[1] == ["mets en marche {area}"]
    lights = LearnedIntent(
        id=7, kind="sentence", intent="HassTurnOn", sentences=("mets en marche {area}",), slots={"domain": "light"}
    )
    assert len(stock_blocks(lights, body, data)[0]) == 1

    merged, used = merge_learned(
        {**data, "intents": {"HassTurnOn": body}}, [by_name, by_area], add_intents=False, home_assistant=True
    )
    assert used == [5]
    lists = build_slot_lists(
        [{"name": "lampe", "domain": "light"}, {"name": "thermomètre", "domain": "sensor"}], ["salon"], []
    )
    parsed = Intents.from_dict(merged)
    match = recognize_best("mets en marche la lampe", parsed, slot_lists=lists, language="fr", best_slot_name="name")
    assert match.intent.name == "HassTurnOn" and match.intent_metadata["learned_id"] == 5
    assert recognize_best("mets en marche le thermomètre", parsed, slot_lists=lists, language="fr") is None


def test_export_keeps_stock_intent_sentences_out_of_the_sentence_file(stores):
    import yaml

    store, _ = stores
    for intent, sentence in (("ReachyMoveHead", "tourne-toi vers [la ]{direction}"), ("HassTurnOn", "<allume> {name}")):
        intents.add_suggestion(store, _row(intent, sentence), [])
    files = custom_sentence_data(PACKAGED_SENTENCES_DIR, "fr")
    exported = intents.export_sentences(store, "fr", set(files["intents"]))
    assert set(exported["sentences"]["intents"]) == {"ReachyMoveHead"}
    assert set(exported["home_assistant"]) == {"HassTurnOn"}

    # The exported file parses next to the packaged ones.
    combined = copy.deepcopy(files)
    from hassil.util import merge_dict

    merge_dict(combined, yaml.safe_load(yaml.safe_dump(exported["sentences"])))
    parsed = Intents.from_dict(combined)
    match = recognize_best("tourne-toi vers la droite", parsed, slot_lists=build_slot_lists([], [], []), language="fr")
    assert match.intent.name == "ReachyMoveHead"


def test_rerun_keeps_the_sentences_the_session_activated(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    config = {**_DEFAULTS, "dream": {"intents": {"auto_use_threshold": 0.9}}}
    llm.answer = ANSWER
    _run(store, artifacts, "s1", config=config)
    (first,) = intents.active_suggestions(store, "fr")

    # The re-run does not see its own active sentence as already matching: it proposes it again.
    progress = _run(store, artifacts, "s1", config=config, progress={"written": True})
    (again,) = intents.active_suggestions(store, "fr")
    assert progress["intents"]["retracted"] >= 1
    assert again["sentences"] == first["sentences"] and again["id"] != first["id"]


def test_correcting_a_sentence_of_a_learned_intent(stores):
    from monitoring.jobs.dream_intents import correction_problems

    store, _ = stores
    intents.add_suggestion(
        store,
        {
            **_row("ReachyFollowMe", "suis-moi"),
            "kind": "intent",
            "tool": {"tool": "head_tracking", "arguments": {"enabled": True}},
            "response": "Je te suis",
        },
        [],
    )
    sentence_id = intents.add_suggestion(store, {**_row("ReachyFollowMe", "regarde-moi"), "status": "proposed"}, [])
    row = intents.get_suggestion(store, sentence_id)
    assert correction_problems(store, row, ["regarde-moi vers [la ]{direction}"], None, None) == []
    assert correction_problems(store, row, ["regarde {nope}"], None, None)


def test_a_learned_sentence_that_breaks_parsing_keeps_the_custom_intents(monkeypatch):
    from examples.shared.intent_engine import matcher

    def broken(data, learned, **kwargs):
        merged, used = merge_learned(data, learned, **kwargs)
        if learned:
            merged["intents"]["ReachyDance"]["data"].append({"sentences": ["danse <undefined_rule>"]})
        return merged, used

    monkeypatch.setattr(matcher, "merge_learned", broken)
    learned = [LearnedIntent(id=1, kind="sentence", intent="ReachyDance", sentences=("bouge",))]
    custom = matcher._load_custom(
        matcher.custom_sentence_data(PACKAGED_SENTENCES_DIR, "fr"), "fr", lambda n: True, learned
    )
    assert custom is not None and "ReachyDance" in custom.intents.intents


def test_stock_set_with_learned_sentences_is_cached():
    from examples.shared.intent_engine import matcher

    allowed = ("HassTurnOn",)
    outside = [LearnedIntent(id=1, kind="sentence", intent="HassTurnOff", sentences=("éteins {name}",))]
    assert matcher._load_stock("fr", allowed, outside) is matcher._load_stock_cached("fr", allowed)
    inside = [LearnedIntent(id=2, kind="sentence", intent="HassTurnOn", sentences=("mets en marche {name}",))]
    first = matcher._load_stock("fr", allowed, inside)
    assert first is matcher._load_stock("fr", allowed, inside)
    assert first is not matcher._load_stock_cached("fr", allowed)


def test_optional_target_slots_fit_several_stock_blocks():
    data = home_assistant_intents.get_intents("fr")
    body = data["intents"]["HassTurnOn"]
    item = LearnedIntent(
        id=3, kind="sentence", intent="HassTurnOn", sentences=("mets en marche [<le>]{name} [<dans> [<le>]{area}]",)
    )
    blocks, misfits = stock_blocks(item, body, data)
    assert not misfits
    shapes = {b["metadata"].get("slot_combination") for b in blocks}
    assert {"name_only", "name_area"} <= shapes


def test_catalog_keeps_an_empty_recorded_allow_list(stores, monkeypatch):
    from monitoring.jobs.dream_intents import DEFAULTS, IntentCatalog

    store, _ = stores
    config = {"language": "fr-FR", "intent_engine": {"enabled": True, "home_assistant": True, "allowed_intents": []}}
    assert IntentCatalog(store, config, dict(DEFAULTS)).allowed == ()


def test_base_language_helpers_agree():
    from examples.shared.intent_engine.matcher import base_language

    for code in ("fr-FR", "FR-fr", "en", " de-CH ", ""):
        assert intents.base_language(code) == base_language(code)


def test_live_uses_only_read_new_events(stores, llm):
    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    llm.answer = ANSWER
    _run(store, artifacts, "s1")
    sid = intents.list_suggestions(store)["suggestions"][0]["id"]

    def use(ts):
        event = {"handled_by": "intent", "intent": "X", "learned": sid}
        store.write_batch(
            [("events", {"session_id": "s1", "turn_idx": 9, "ts": ts, "kind": "intent_engine", "data": event})]
        )

    use(1300.0)
    assert intents.live_uses(store, [sid]) == {sid: 1}
    assert intents.live_uses(store, [sid]) == {sid: 1}  # counted once
    use(1301.0)
    assert intents.list_suggestions(store)["suggestions"][0]["live_uses"] == 2


def test_correcting_a_custom_sentence_next_to_a_learned_home_assistant_one(stores):
    from monitoring.jobs.dream_intents import correction_problems

    store, _ = stores
    intents.add_suggestion(store, _row("HassTurnOn", "mets en marche [<le>]{name}"), [])
    sentence_id = intents.add_suggestion(
        store, {**_row("ReachyMoveHead", "pivote vers [la ]{direction}"), "status": "proposed"}, []
    )
    row = intents.get_suggestion(store, sentence_id)
    assert correction_problems(store, row, ["tourne-toi vers [la ]{direction}"], None, None) == []


def test_renaming_a_learned_intent_keeps_its_sentences(stores):
    store, _ = stores
    intent_id = intents.add_suggestion(
        store,
        {**_row("LookAround", "regarde autour"), "kind": "intent", "tool": {"tool": "head_tracking", "arguments": {}}},
        [],
    )
    sentence_id = intents.add_suggestion(store, _row("LookAround", "jette un œil"), [])
    renamed = intents.review_suggestion(store, intent_id, "correct", reviewer="ann", intent="LookEverywhere")
    assert renamed["intent"] == "LookEverywhere"
    assert intents.get_suggestion(store, sentence_id)["intent"] == "LookEverywhere"


def test_one_bad_learned_suggestion_does_not_drop_the_others():
    # hassil only reports a missing rule when matching reaches it, and the live
    # engine then skips the whole sentence set: the rules a sentence uses are checked too.
    data = {"language": "fr", "intents": {"A": {"data": [{"sentences": ["a"]}]}}, "expansion_rules": {"r": "x <nope>"}}
    good = LearnedIntent(id=1, kind="sentence", intent="A", sentences=("b",))
    bad = LearnedIntent(id=2, kind="sentence", intent="A", sentences=("c <r>",))
    merged, used = merge_learned(data, [good, bad], add_intents=False)
    assert used == [1]
    assert "unknown rule <nope>" in sentence_problems("c <r>", data)


def test_concurrent_reviews_of_one_suggestion(stores, monkeypatch):
    store, _ = stores
    suggestion_id = intents.add_suggestion(store, {**_row("ReachyDance", "bouge"), "status": "proposed"}, [])
    stale = intents.get_suggestion(store, suggestion_id)
    intents.review_suggestion(store, suggestion_id, "correct", reviewer="ann", sentences=["bouge un peu"])
    # The second reviewer read the row before the first one saved.
    monkeypatch.setattr(intents, "get_suggestion", lambda *_: dict(stale))
    with pytest.raises(ValueError, match="meanwhile"):
        intents.review_suggestion(store, suggestion_id, "correct", reviewer="bob", sentences=["bouge encore"])
    monkeypatch.undo()
    assert [r["sentences"] for r in intents.active_suggestions(store, "fr")] == [["bouge un peu"]]


def test_listing_suggestions_does_not_write(stores, llm):
    from monitoring import schema

    store, artifacts = stores
    _session(store, "s1", llm.base_url)
    llm.answer = ANSWER
    _run(store, artifacts, "s1")
    listed = intents.list_suggestions(store)
    with store.engine.connect() as conn:
        assert conn.execute(schema.intent_use_scan.select()).first() is None
    assert listed["languages"] == []  # nothing active yet
    intents.review_suggestion(store, listed["suggestions"][0]["id"], "approve", reviewer="ann")
    assert intents.list_suggestions(store)["languages"] == ["fr"]


def test_export_header_names_the_configured_directory(api, tmp_path, monkeypatch):
    import shutil

    client, store = api
    custom = tmp_path / "my-sentences"
    shutil.copytree(PACKAGED_SENTENCES_DIR, custom)
    monkeypatch.setenv("INTENT_ENGINE_SENTENCES_DIR", str(custom))
    text = client.get("/api/review/intents/export", params={"language": "fr"}).text
    assert f"add to {custom / 'fr'}/" in text and f"add to {custom / 'client_tools.yaml'}" in text

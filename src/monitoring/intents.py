# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learned intent sentences: suggestions of the ``dream`` job and their review.

The ``dream`` job reads what people said in turns the intent engine did not
answer and proposes:

- ``sentence``: new hassil sentences for an existing intent, so the engine
  understands the way people actually phrase a command;
- ``intent``: a new intent running a client tool the LLM was asked to call for a
  recurring request, so the request skips the LLM next time.

A suggestion stays ``proposed`` until a reviewer approves it (sentences can be
used right away above an optional confidence threshold). ``active`` suggestions
are merged into the live intent engine at session start. Evidence from several
sessions accumulates on one suggestion, so recurring requests rise to the top.
Rejecting, correcting and superseding never delete rows.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import delete, func, insert, select, update

from monitoring import schema
from monitoring.store import SessionStore

KINDS = ("sentence", "intent")
USABLE_STATUSES = ("proposed", "active")
REVIEW_ACTIONS = ("approve", "correct", "reject", "restore")
DREAM_PREFIX = "dream:"
HUMAN_PREFIX = "human:"
INTENT_EVENT = "intent_engine"


def _rows(store: SessionStore, stmt) -> list[dict[str, Any]]:
    with store.engine.connect() as conn:
        return [dict(r) for r in conn.execute(stmt).mappings()]


def base_language(code: str | None) -> str:
    """``fr-FR`` -> ``fr``."""
    return (code or "").split("-")[0].strip().lower()


def sentence_key(sentence: str) -> str:
    """Comparison key of a sentence template (case and spacing do not matter)."""
    return " ".join(str(sentence).split()).casefold()


def same_definition(row: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Whether ``candidate`` proposes what ``row`` already holds.

    Sentences: same intent and fixed slots, sharing a sentence. Intents: same tool and arguments.
    """
    if row["kind"] != candidate["kind"] or row["language"] != candidate["language"]:
        return False
    if row["kind"] == "intent":
        return (row.get("tool") or {}) == (candidate.get("tool") or {})
    if row["intent"] != candidate["intent"] or (row.get("slots") or {}) != (candidate.get("slots") or {}):
        return False
    keys = {sentence_key(s) for s in row.get("sentences") or []}
    return any(sentence_key(s) in keys for s in candidate.get("sentences") or [])


def is_open(row: dict[str, Any]) -> bool:
    """Whether a suggestion is proposed and not reviewed: its sentences may still change.

    An active suggestion is in use (approved, or auto-used above the threshold), so
    new sentences become a suggestion of their own instead.
    """
    return row["status"] == "proposed" and row["reviewed_at"] is None


# ------------------------------------------------------------------ reads
def get_suggestion(store: SessionStore, suggestion_id: int) -> dict[str, Any] | None:
    """One suggestion row, or None."""
    rows = _rows(store, select(schema.intent_suggestions).where(schema.intent_suggestions.c.id == suggestion_id))
    return rows[0] if rows else None


def suggestions_for(store: SessionStore, language: str, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
    """Suggestions of a base language with one of ``statuses``, oldest first."""
    s = schema.intent_suggestions
    stmt = select(s).where(s.c.language == base_language(language), s.c.status.in_(statuses)).order_by(s.c.id)
    return _rows(store, stmt)


def active_suggestions(store: SessionStore, language: str) -> list[dict[str, Any]]:
    """Suggestions the live intent engine loads for a session language."""
    return suggestions_for(store, language, ("active",))


def evidence_turns(store: SessionStore, suggestion_ids: list[int]) -> dict[int, set[tuple[str, int]]]:
    """``{suggestion id: {(session_id, turn_idx)}}``."""
    if not suggestion_ids:
        return {}
    e = schema.intent_evidence
    out: dict[int, set[tuple[str, int]]] = defaultdict(set)
    stmt = select(e.c.suggestion_id, e.c.session_id, e.c.turn_idx).where(e.c.suggestion_id.in_(suggestion_ids))
    with store.engine.connect() as conn:
        for row in conn.execute(stmt):
            out[row.suggestion_id].add((row.session_id, row.turn_idx))
    return out


# ----------------------------------------------------------------- writes
def add_suggestion(store: SessionStore, row: dict[str, Any], evidence: list[dict[str, Any]]) -> int:
    """Insert a suggestion and its evidence (``session_id``, ``turn_idx``, ``text``, ``tool_calls``)."""
    now = time.time()
    with store.engine.begin() as conn:
        suggestion_id = conn.execute(
            insert(schema.intent_suggestions).values(**{"created_at": now, "updated_at": now, **row})
        ).inserted_primary_key[0]
        if evidence:
            conn.execute(insert(schema.intent_evidence), [{**e, "suggestion_id": suggestion_id} for e in evidence])
    return int(suggestion_id)


def add_evidence(
    store: SessionStore, suggestion_id: int, evidence: list[dict[str, Any]], *, sentences: list[str] | None = None
) -> int:
    """Attach more evidence (turns already attached are skipped); returns how many were added.

    ``sentences`` are merged into a suggestion still waiting for review (``is_open``);
    any other one keeps its sentences.
    """
    existing = evidence_turns(store, [suggestion_id]).get(suggestion_id, set())
    new = [e for e in evidence if (e["session_id"], e["turn_idx"]) not in existing]
    row = get_suggestion(store, suggestion_id) or {}
    values: dict[str, Any] = {}
    if sentences and row and is_open(row):
        current = list(row["sentences"] or [])
        keys = {sentence_key(s) for s in current}
        extra = [s for s in sentences if sentence_key(s) not in keys]
        if extra:
            values["sentences"] = current + extra
    with store.engine.begin() as conn:
        if new:
            conn.execute(insert(schema.intent_evidence), [{**e, "suggestion_id": suggestion_id} for e in new])
        if new or values:
            _set(conn, suggestion_id, **values)
    return len(new)


def _set(conn, suggestion_id: int, **values: Any) -> None:
    conn.execute(
        update(schema.intent_suggestions)
        .where(schema.intent_suggestions.c.id == suggestion_id)
        .values(updated_at=time.time(), **values)
    )


def retract_session_suggestions(store: SessionStore, session_id: str) -> int:
    """Undo a previous ``dream`` run over ``session_id`` (before a re-run).

    The session's evidence is removed from unreviewed dream suggestions; those left
    without evidence are superseded. Returns how many were superseded.
    """
    s, e = schema.intent_suggestions, schema.intent_evidence
    candidates = _rows(
        store,
        select(s.c.id)
        .join(e, e.c.suggestion_id == s.c.id)
        .where(
            e.c.session_id == session_id,
            s.c.status.in_(USABLE_STATUSES),
            s.c.reviewed_at.is_(None),
            s.c.source.like(DREAM_PREFIX + "%"),
        )
        .distinct(),
    )
    retracted = 0
    with store.engine.begin() as conn:
        for row in candidates:
            conn.execute(delete(e).where(e.c.suggestion_id == row["id"], e.c.session_id == session_id))
            left = conn.execute(select(func.count()).select_from(e).where(e.c.suggestion_id == row["id"])).scalar()
            if not left:
                _set(conn, row["id"], status="superseded", superseded_by=None)
                retracted += 1
    return retracted


def review_suggestion(
    store: SessionStore,
    suggestion_id: int,
    action: str,
    *,
    reviewer: str,
    sentences: list[str] | None = None,
    intent: str | None = None,
    response: str | None = None,
) -> dict[str, Any]:
    """Apply a review action; returns the resulting suggestion (the new row for ``correct``).

    - ``approve``: the suggestion becomes ``active`` (loaded by new sessions);
    - ``correct``: a ``human:<reviewer>`` copy with new sentences (and, for a new
      intent, name and response) replaces it and is ``active``;
    - ``reject``: never used, and the dream job does not propose it again;
    - ``restore``: a rejected suggestion becomes ``active``.
    """
    row = get_suggestion(store, suggestion_id)
    if row is None:
        raise KeyError(suggestion_id)
    if action not in REVIEW_ACTIONS:
        raise ValueError(action)
    now = time.time()
    source = HUMAN_PREFIX + reviewer
    if action in ("approve", "reject"):
        if row["status"] not in USABLE_STATUSES:
            raise ValueError(f"a {row['status']} suggestion cannot be reviewed")
        status = "active" if action == "approve" else "rejected"
        with store.engine.begin() as conn:
            _set(conn, suggestion_id, status=status, reviewed_by=source, reviewed_at=now)
        return get_suggestion(store, suggestion_id) or row
    if action == "restore":
        if row["status"] != "rejected":
            raise ValueError("only rejected suggestions can be restored")
        with store.engine.begin() as conn:
            _set(conn, suggestion_id, status="active", reviewed_by=source, reviewed_at=now)
        return get_suggestion(store, suggestion_id) or row
    if row["status"] not in USABLE_STATUSES:
        raise ValueError(f"a {row['status']} suggestion cannot be corrected")
    cleaned = [" ".join(s.split()) for s in (sentences if sentences is not None else row["sentences"] or [])]
    cleaned = [s for s in cleaned if s]
    if not cleaned:
        raise ValueError("at least one sentence is required")
    evidence = _rows(
        store,
        select(schema.intent_evidence.c.session_id, schema.intent_evidence.c.turn_idx)
        .add_columns(schema.intent_evidence.c.text, schema.intent_evidence.c.tool_calls)
        .where(schema.intent_evidence.c.suggestion_id == suggestion_id)
        .order_by(schema.intent_evidence.c.id),
    )
    new_id = add_suggestion(
        store,
        {
            "kind": row["kind"],
            "language": row["language"],
            "intent": (intent or row["intent"]).strip() if row["kind"] == "intent" else row["intent"],
            "sentences": cleaned,
            "slots": row["slots"],
            "tool": row["tool"],
            "response": (response if response is not None else row["response"]) if row["kind"] == "intent" else None,
            "confidence": 1.0,
            # The reviewer's sentences are not re-checked against the evidence.
            "validated": False,
            "validation": {"notes": [f"corrected by {reviewer}"]},
            "status": "active",
            "source": source,
            "reviewed_by": source,
            "reviewed_at": now,
        },
        evidence,
    )
    with store.engine.begin() as conn:
        _set(conn, suggestion_id, status="superseded", superseded_by=new_id, reviewed_by=source, reviewed_at=now)
    return get_suggestion(store, new_id) or {}


# ----------------------------------------------------------------- review
def list_suggestions(
    store: SessionStore,
    *,
    status: str = "open",
    language: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Suggestions with evidence and live uses; the most recurrent first.

    ``status``: ``open`` (usable and not reviewed yet), ``usable`` (proposed + active),
    one concrete status, or ``all``.
    """
    s = schema.intent_suggestions
    conditions = []
    if language:
        conditions.append(s.c.language == base_language(language))
    if kind:
        conditions.append(s.c.kind == kind)
    if status == "open":
        conditions += [s.c.status.in_(USABLE_STATUSES), s.c.reviewed_at.is_(None)]
    elif status == "usable":
        conditions.append(s.c.status.in_(USABLE_STATUSES))
    elif status != "all":
        conditions.append(s.c.status == status)
    e = schema.intent_evidence
    # Evidence rows are unique per (suggestion, session, turn): their count is the occurrence count.
    occurrences = select(e.c.suggestion_id, func.count().label("n")).group_by(e.c.suggestion_id).subquery()
    stmt = (
        select(s)
        .outerjoin(occurrences, occurrences.c.suggestion_id == s.c.id)
        .where(*conditions)
        .order_by(func.coalesce(occurrences.c.n, 0).desc(), s.c.created_at.desc(), s.c.id.desc())
        .limit(limit)
        .offset(offset)
    )
    with store.engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(s).where(*conditions)).scalar_one()
    return {"total": total, "suggestions": attach_details(store, _rows(store, stmt))}


def open_suggestion_count(store: SessionStore) -> int:
    """Suggestions waiting for a first review."""
    s = schema.intent_suggestions
    with store.engine.connect() as conn:
        return conn.execute(
            select(func.count()).select_from(s).where(s.c.status.in_(USABLE_STATUSES), s.c.reviewed_at.is_(None))
        ).scalar_one()


def live_uses(store: SessionStore, suggestion_ids: list[int]) -> Counter:
    """How many live turns each suggestion answered (intent engine events tagged ``learned``)."""
    wanted = set(suggestion_ids)
    uses: Counter = Counter()
    if not wanted:
        return uses
    ev = schema.events
    learned = ev.c.data["learned"].as_integer()
    stmt = (
        select(learned, func.count())
        .where(ev.c.kind == INTENT_EVENT, ev.c.data["handled_by"].as_string() == "intent", learned.in_(wanted))
        .group_by(learned)
    )
    with store.engine.connect() as conn:
        for learned_id, count in conn.execute(stmt):
            uses[int(learned_id)] = int(count)
    return uses


def attach_details(store: SessionStore, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add evidence (with audio), occurrence and live-use counts."""
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    e = schema.intent_evidence
    evidence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(store, select(e).where(e.c.suggestion_id.in_(ids)).order_by(e.c.id)):
        evidence[row["suggestion_id"]].append(row)
    audio: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sid in {ev["session_id"] for evs in evidence.values() for ev in evs}:
        for media in store.rows("media", sid, modality="audio_user"):
            audio[(sid, media["turn_idx"])].append(
                {"key": media["artifact_key"], "duration_secs": media["duration_secs"]}
            )
    uses = live_uses(store, ids)
    out = []
    for row in rows:
        evs = evidence.get(row["id"], [])
        out.append(
            {
                **row,
                "occurrences": len({(ev["session_id"], ev["turn_idx"]) for ev in evs}),
                "sessions": len({ev["session_id"] for ev in evs}),
                "live_uses": uses.get(row["id"], 0),
                "evidence": [{**ev, "audio": audio.get((ev["session_id"], ev["turn_idx"]), [])} for ev in evs],
            }
        )
    return out


def export_sentences(store: SessionStore, language: str, custom_intents: set[str]) -> dict[str, Any]:
    """Active suggestions as sentence-file and ``client_tools.yaml`` content, to commit them to the repository.

    Only sentences of ``custom_intents`` (defined by the custom sentence files) and
    learned intents go to the sentence file. Sentences of stock Home Assistant
    intents are returned under ``home_assistant``: a custom file defining a stock
    intent would take it over without the stock lists, rules and context.
    """
    sentences: dict[str, Any] = {"language": base_language(language), "intents": {}}
    responses: dict[str, Any] = {}
    client_tools: dict[str, Any] = {}
    home_assistant: dict[str, Any] = {}
    rows = active_suggestions(store, language)
    custom = set(custom_intents) | {row["intent"] for row in rows if row["kind"] == "intent"}
    for row in rows:
        block: dict[str, Any] = {"sentences": list(row["sentences"] or [])}
        if row["slots"]:
            block["slots"] = row["slots"]
        target = sentences["intents"] if row["intent"] in custom else home_assistant
        target.setdefault(row["intent"], {"data": []})["data"].append(block)
        if row["kind"] == "intent":
            responses[row["intent"]] = {"default": row["response"] or ""}
            client_tools[row["intent"]] = row["tool"] or {}
    if responses:
        sentences["responses"] = {"intents": responses}
    return {"sentences": sentences, "client_tools": {"intents": client_tools}, "home_assistant": home_assistant}

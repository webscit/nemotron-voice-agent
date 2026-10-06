# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""People, speaker attribution and memories.

- **People**: who the agent talks to. A live session is attributed to the person
  picked in the client; review can reassign a whole session or single turns, and
  merge two people found to be the same one.
- **Memories**: facts about a person extracted offline by the ``dream`` job. A
  memory above the confidence threshold is ``active`` right away (used in live
  prompts) and reviewed afterwards; below it, it stays ``proposed`` until a
  reviewer approves it. Forgetting, correcting and superseding never delete rows.
"""

from __future__ import annotations

import functools
import time
import uuid
from collections import defaultdict
from typing import Any

from sqlalchemy import delete, func, insert, select, update

from monitoring import schema
from monitoring.store import SessionStore

SESSION_TURN = -1
USABLE_STATUSES = ("proposed", "active")
REVIEW_ACTIONS = ("approve", "correct", "forget")
# ``kv`` key of a merged person, pointing at the person they were merged into.
PERSON_ALIAS_PREFIX = "person_alias:"
_MAX_ALIAS_HOPS = 16


def _rows(store: SessionStore, stmt) -> list[dict[str, Any]]:
    with store.engine.connect() as conn:
        return [dict(r) for r in conn.execute(stmt).mappings()]


# ------------------------------------------------------------------- people
def list_people(store: SessionStore, *, include_archived: bool = False) -> list[dict[str, Any]]:
    """People sorted by name."""
    p = schema.people
    stmt = select(p).order_by(func.lower(p.c.name))
    if not include_archived:
        stmt = stmt.where(p.c.archived.is_(False))
    return _rows(store, stmt)


def get_person(store: SessionStore, person_id: str) -> dict[str, Any] | None:
    """One person, or None."""
    rows = _rows(store, select(schema.people).where(schema.people.c.id == person_id))
    return rows[0] if rows else None


def create_person(store: SessionStore, name: str) -> dict[str, Any]:
    """Add a person."""
    row = {"id": uuid.uuid4().hex[:12], "name": name.strip(), "created_at": time.time(), "archived": False}
    with store.engine.begin() as conn:
        conn.execute(insert(schema.people).values(**row))
    return row


def resolve_person_id(store: SessionStore, person_id: str) -> str:
    """The id ``person_id`` was merged into, or ``person_id`` itself.

    Live clients cache the speaker gallery for a whole session, so they keep
    reporting a merged person's old id until they reconnect.
    """
    kv = schema.kv
    with store.engine.connect() as conn:
        for _ in range(_MAX_ALIAS_HOPS):
            alias = conn.execute(select(kv.c.value).where(kv.c.key == PERSON_ALIAS_PREFIX + person_id)).scalar()
            if not isinstance(alias, dict) or not alias.get("person_id"):
                break
            person_id = alias["person_id"]
    return person_id


def merge_people(store: SessionStore, source_id: str, target_id: str, *, annotator: str) -> dict[str, Any]:
    """Move everything known about ``source_id`` to ``target_id``, then delete ``source_id``.

    Turn attributions, voice and face embeddings and memories move; the target
    keeps its name. Raises ``KeyError`` for an unknown person and ``ValueError``
    for a self-merge or an archived target.
    """
    if source_id == target_id:
        raise ValueError("cannot merge a person into themselves")
    source, target = get_person(store, source_id), get_person(store, target_id)
    if source is None:
        raise KeyError(source_id)
    if target is None:
        raise KeyError(target_id)
    if target["archived"]:
        raise ValueError("unarchive the person to keep before merging into them")
    moved: dict[str, int] = {}
    kv = schema.kv
    now = time.time()
    with store.engine.begin() as conn:
        for name, table in (
            ("turns", schema.speaker_assignments),
            ("embeddings", schema.voice_embeddings),
            ("memories", schema.memories),
        ):
            result = conn.execute(update(table).where(table.c.person_id == source_id).values(person_id=target_id))
            moved[name] = result.rowcount or 0
        # Earlier merges into the source now point at the target directly.
        for row in conn.execute(select(kv.c.key, kv.c.value).where(kv.c.key.like(PERSON_ALIAS_PREFIX + "%"))).all():
            if isinstance(row.value, dict) and row.value.get("person_id") == source_id:
                conn.execute(update(kv).where(kv.c.key == row.key).values(value={**row.value, "person_id": target_id}))
        alias = {"person_id": target_id, "name": source["name"], "merged_by": annotator, "merged_at": now}
        conn.execute(delete(kv).where(kv.c.key == PERSON_ALIAS_PREFIX + source_id))
        conn.execute(insert(kv).values(key=PERSON_ALIAS_PREFIX + source_id, value=alias, updated_at=now))
        conn.execute(delete(schema.people).where(schema.people.c.id == source_id))
    return {"person": target, "merged": source, "moved": moved}


def find_person_by_name(store: SessionStore, name: str) -> dict[str, Any] | None:
    """The oldest non-archived person with this name (case-insensitive), or None."""
    # Compared in Python: SQLite's ``lower()`` only folds ASCII ("Élodie" != "élodie").
    wanted = name.strip().casefold()
    matches = [p for p in list_people(store) if p["name"].strip().casefold() == wanted]
    return min(matches, key=lambda p: p["created_at"]) if matches else None


def find_or_create_person(store: SessionStore, name: str) -> dict[str, Any]:
    """The non-archived person named ``name`` (case-insensitive), created when missing."""
    return find_person_by_name(store, name) or create_person(store, name)


def update_person(
    store: SessionStore, person_id: str, *, name: str | None = None, archived: bool | None = None
) -> dict[str, Any]:
    """Rename or (un)archive a person; raises ``KeyError`` when unknown."""
    values: dict[str, Any] = {}
    if name is not None:
        values["name"] = name.strip()
    if archived is not None:
        values["archived"] = archived
    with store.engine.begin() as conn:
        if values:
            conn.execute(update(schema.people).where(schema.people.c.id == person_id).values(**values))
    person = get_person(store, person_id)
    if person is None:
        raise KeyError(person_id)
    return person


# --------------------------------------------------------------- speakers
def assign_speaker(
    store: SessionStore, session_id: str, person_id: str | None, *, source: str, turn_idx: int = SESSION_TURN
) -> None:
    """Attribute a session (``turn_idx=-1``) or one turn to a person; ``None`` clears it."""
    if person_id:
        person_id = resolve_person_id(store, person_id)
    sa = schema.speaker_assignments
    where = (sa.c.session_id == session_id) & (sa.c.turn_idx == turn_idx)
    with store.engine.begin() as conn:
        conn.execute(delete(sa).where(where))
        if person_id:
            conn.execute(
                insert(sa).values(
                    session_id=session_id,
                    turn_idx=turn_idx,
                    person_id=person_id,
                    source=source,
                    created_at=time.time(),
                )
            )


def speakers_for(store: SessionStore, session_id: str) -> dict[str, Any]:
    """``{"session": person_id | None, "turns": {turn_idx: person_id}}`` (explicit rows only)."""
    sa = schema.speaker_assignments
    out: dict[str, Any] = {"session": None, "turns": {}}
    for row in _rows(store, select(sa).where(sa.c.session_id == session_id)):
        if row["turn_idx"] == SESSION_TURN:
            out["session"] = row["person_id"]
        else:
            out["turns"][row["turn_idx"]] = row["person_id"]
    return out


def resolve_speaker(speakers: dict[str, Any], turn_idx: int) -> str | None:
    """Person speaking in ``turn_idx``: the turn override, else the session person."""
    return speakers["turns"].get(turn_idx, speakers["session"])


# --------------------------------------------------------------- memories
def active_memories(store: SessionStore, person_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Memories used in live prompts for ``person_id``: most confident, then most recent."""
    m = schema.memories
    stmt = (
        select(m)
        .where(m.c.person_id == person_id, m.c.status == "active")
        .order_by(m.c.confidence.desc(), m.c.updated_at.desc())
        .limit(limit)
    )
    return _rows(store, stmt)


def usable_memories(store: SessionStore, person_ids: list[str]) -> list[dict[str, Any]]:
    """``proposed`` and ``active`` memories of these people (what the ``dream`` job builds on)."""
    if not person_ids:
        return []
    m = schema.memories
    return _rows(
        store, select(m).where(m.c.person_id.in_(person_ids), m.c.status.in_(USABLE_STATUSES)).order_by(m.c.id)
    )


def add_memory(store: SessionStore, row: dict[str, Any], evidence: list[dict[str, Any]]) -> int:
    """Insert a memory and its evidence (``session_id``, ``turn_idx``, ``quote``); returns its id."""
    now = time.time()
    with store.engine.begin() as conn:
        memory_id = conn.execute(
            insert(schema.memories).values(**{"created_at": now, "updated_at": now, **row})
        ).inserted_primary_key[0]
        if evidence:
            conn.execute(insert(schema.memory_evidence), [{**e, "memory_id": memory_id} for e in evidence])
    return int(memory_id)


def _set_memory(conn, memory_id: int, **values: Any) -> None:
    conn.execute(
        update(schema.memories).where(schema.memories.c.id == memory_id).values(updated_at=time.time(), **values)
    )


def get_memory(store: SessionStore, memory_id: int) -> dict[str, Any] | None:
    """One memory row, or None."""
    rows = _rows(store, select(schema.memories).where(schema.memories.c.id == memory_id))
    return rows[0] if rows else None


def supersede(store: SessionStore, old_id: int, new_id: int) -> None:
    """Mark ``old_id`` as replaced by ``new_id``."""
    with store.engine.begin() as conn:
        _set_memory(conn, old_id, status="superseded", superseded_by=new_id)


def review_memory(
    store: SessionStore,
    memory_id: int,
    action: str,
    *,
    reviewer: str,
    text: str | None = None,
    person_id: str | None = None,
) -> dict[str, Any]:
    """Apply a review action; returns the resulting memory (the new row for ``correct``).

    - ``approve``: the memory becomes ``active`` (and replaces what it supersedes);
    - ``correct``: a ``human:<reviewer>`` copy with the new text / person replaces it;
    - ``forget``: the memory is never used again.
    """
    memory = get_memory(store, memory_id)
    if memory is None:
        raise KeyError(memory_id)
    if action not in REVIEW_ACTIONS:
        raise ValueError(action)
    now = time.time()
    source = f"human:{reviewer}"
    if action == "forget":
        with store.engine.begin() as conn:
            _set_memory(conn, memory_id, status="forgotten", reviewed_by=source, reviewed_at=now)
        return get_memory(store, memory_id) or memory
    if action == "approve":
        with store.engine.begin() as conn:
            _set_memory(conn, memory_id, status="active", reviewed_by=source, reviewed_at=now)
            if memory["supersedes"]:
                _set_memory(conn, memory["supersedes"], status="superseded", superseded_by=memory_id)
        return get_memory(store, memory_id) or memory
    evidence = _rows(
        store,
        select(schema.memory_evidence.c.session_id, schema.memory_evidence.c.turn_idx, schema.memory_evidence.c.quote)
        .where(schema.memory_evidence.c.memory_id == memory_id)
        .order_by(schema.memory_evidence.c.id),
    )
    new_id = add_memory(
        store,
        {
            "person_id": person_id or memory["person_id"],
            "text": (text or memory["text"]).strip(),
            "category": memory["category"],
            "language": memory["language"],
            "confidence": 1.0,
            "status": "active",
            "source": source,
            "supersedes": memory_id,
            "reviewed_by": source,
            "reviewed_at": now,
        },
        evidence,
    )
    with store.engine.begin() as conn:
        _set_memory(conn, memory_id, status="superseded", superseded_by=new_id, reviewed_by=source, reviewed_at=now)
        if memory["supersedes"]:
            _set_memory(conn, memory["supersedes"], status="superseded", superseded_by=new_id)
    return get_memory(store, new_id) or {}


def retract_session_memories(store: SessionStore, session_id: str) -> int:
    """Supersede unreviewed memories whose only evidence is ``session_id`` (before a ``dream`` re-run)."""
    m, e = schema.memories, schema.memory_evidence
    candidates = _rows(
        store,
        select(m.c.id)
        .join(e, e.c.memory_id == m.c.id)
        .where(e.c.session_id == session_id, m.c.status.in_(USABLE_STATUSES), m.c.reviewed_at.is_(None))
        .distinct(),
    )
    retracted = 0
    with store.engine.begin() as conn:
        for row in candidates:
            sessions = {r[0] for r in conn.execute(select(e.c.session_id).where(e.c.memory_id == row["id"]).distinct())}
            if sessions == {session_id}:
                _set_memory(conn, row["id"], status="superseded", superseded_by=None)
                retracted += 1
    return retracted


def record_uses(store: SessionStore, session_id: str, memory_ids: list[int]) -> None:
    """Remember which memories were injected into a live session prompt."""
    if not memory_ids:
        return
    now = time.time()
    with store.engine.begin() as conn:
        conn.execute(
            store._insert(schema.memory_uses)
            .values([{"memory_id": mid, "session_id": session_id, "ts": now} for mid in memory_ids])
            .on_conflict_do_nothing()
        )


def list_memories(
    store: SessionStore,
    *,
    person_id: str | None = None,
    status: str = "open",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Memories with evidence and usage, newest first.

    ``status``: ``open`` (usable and not reviewed yet), ``usable`` (proposed + active),
    one concrete status, or ``all``.
    """
    m = schema.memories
    conditions = []
    if person_id:
        conditions.append(m.c.person_id == person_id)
    if status == "open":
        conditions += [m.c.status.in_(USABLE_STATUSES), m.c.reviewed_at.is_(None)]
    elif status == "usable":
        conditions.append(m.c.status.in_(USABLE_STATUSES))
    elif status != "all":
        conditions.append(m.c.status == status)
    with store.engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(m).where(*conditions)).scalar_one()
    stmt = select(m).where(*conditions).order_by(m.c.created_at.desc(), m.c.id.desc())
    rows = _rows(store, stmt.limit(limit).offset(offset))
    return {"total": total, "memories": attach_details(store, rows)}


def open_memory_count(store: SessionStore) -> int:
    """Memories waiting for a first review."""
    m = schema.memories
    with store.engine.connect() as conn:
        return conn.execute(
            select(func.count()).select_from(m).where(m.c.status.in_(USABLE_STATUSES), m.c.reviewed_at.is_(None))
        ).scalar_one()


def attach_details(store: SessionStore, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add evidence (with turn text), usage counts, superseded text and person name."""
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    e, u, t = schema.memory_evidence, schema.memory_uses, schema.turns
    evidence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(store, select(e).where(e.c.memory_id.in_(ids)).order_by(e.c.id)):
        evidence[row["memory_id"]].append(row)
    turn_keys = {(ev["session_id"], ev["turn_idx"]) for evs in evidence.values() for ev in evs}
    turn_text: dict[tuple[str, int], dict[str, Any]] = {}
    for sid in {k[0] for k in turn_keys}:
        for turn in _rows(store, select(t).where(t.c.session_id == sid)):
            turn_text[(sid, turn["idx"])] = {"user_text": turn["user_text"], "bot_text": turn["bot_text"]}
    audio: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sid in {k[0] for k in turn_keys}:
        for media in store.rows("media", sid, modality="audio_user"):
            clip = {"key": media["artifact_key"], "duration_secs": media["duration_secs"]}
            audio[(sid, media["turn_idx"])].append(clip)

    use_sessions: dict[int, set[str]] = defaultdict(set)
    for row in _rows(store, select(u.c.memory_id, u.c.session_id).where(u.c.memory_id.in_(ids))):
        use_sessions[row["memory_id"]].add(row["session_id"])
    all_sessions = sorted({s for sessions in use_sessions.values() for s in sessions})
    replies: dict[str, int] = {}
    if all_sessions:
        with store.engine.connect() as conn:
            replies = dict(
                conn.execute(
                    select(t.c.session_id, func.count())
                    .where(t.c.session_id.in_(all_sessions), t.c.bot_text.is_not(None), t.c.bot_text != "")
                    .group_by(t.c.session_id)
                ).all()
            )
    related = {r["supersedes"] for r in rows if r["supersedes"]}
    superseded_text = {
        r["id"]: r["text"] for r in _rows(store, select(schema.memories).where(schema.memories.c.id.in_(related)))
    }
    names = {p["id"]: p["name"] for p in list_people(store, include_archived=True)}
    out = []
    for row in rows:
        sessions = use_sessions.get(row["id"], set())
        out.append(
            {
                **row,
                "person_name": names.get(row["person_id"]),
                "supersedes_text": superseded_text.get(row["supersedes"]),
                "evidence": [
                    {
                        **ev,
                        **turn_text.get((ev["session_id"], ev["turn_idx"]), {}),
                        "audio": audio.get((ev["session_id"], ev["turn_idx"]), []),
                    }
                    for ev in evidence.get(row["id"], [])
                ],
                "used_in_sessions": len(sessions),
                "used_in_replies": sum(replies.get(s, 0) for s in sessions),
            }
        )
    return out


def person_summaries(store: SessionStore, *, include_archived: bool = False) -> list[dict[str, Any]]:
    """People with their session, memory and voice-ID turn counts."""
    people = list_people(store, include_archived=include_archived)
    sa, m = schema.speaker_assignments, schema.memories
    with store.engine.connect() as conn:
        sessions = dict(
            conn.execute(
                select(sa.c.person_id, func.count(func.distinct(sa.c.session_id))).group_by(sa.c.person_id)
            ).all()
        )
        voice_turns: dict[str, dict[str, int]] = defaultdict(lambda: {"turns": 0, "verified": 0})
        for pid, source, count in conn.execute(
            select(sa.c.person_id, sa.c.source, func.count())
            .where(sa.c.source.like("live:voice-id:%"), sa.c.turn_idx != SESSION_TURN)
            .group_by(sa.c.person_id, sa.c.source)
        ).all():
            voice_turns[pid]["turns"] += count
            if source.endswith(":verified"):
                voice_turns[pid]["verified"] += count
        memory_counts: dict[str, dict[str, int]] = defaultdict(dict)
        for pid, status, count in conn.execute(
            select(m.c.person_id, m.c.status, func.count()).group_by(m.c.person_id, m.c.status)
        ).all():
            memory_counts[pid][status] = count
    return [
        {
            **p,
            "sessions": sessions.get(p["id"], 0),
            "memories": memory_counts.get(p["id"], {}),
            "voice_id_turns": dict(voice_turns.get(p["id"], {"turns": 0, "verified": 0})),
        }
        for p in people
    ]


# ------------------------------------------------------------- live prompt
@functools.lru_cache(maxsize=4)
def _store_for(db_url: str) -> SessionStore:
    store = SessionStore(db_url)
    store.create_schema()
    return store


def live_store() -> SessionStore | None:
    """The session store of the running server, or None when monitoring is off."""
    from monitoring.config import load_monitoring_config

    settings = load_monitoring_config()
    return _store_for(settings.db_url) if settings.enabled else None


def format_memories(memories: list[dict[str, Any]]) -> str:
    """One line per memory, for the system prompt."""
    return "\n".join(f"- {m['text']}" for m in memories) or "- (nothing yet)"

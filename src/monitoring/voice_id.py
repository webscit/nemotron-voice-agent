# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice-ID gallery: speaker embeddings per person and the gallery HTTP API.

The live client computes the embeddings and matches speakers itself; the server
only stores the vectors bound to a person at enrollment and serves one centroid
per person. The wire format is described in ``docs/voice-id-protocol.md``:
L2-normalised little-endian ``float32`` vectors, base64-encoded, comparable only
within one opaque *model key*.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import delete, func, insert, select

from monitoring import memories, schema
from monitoring.store import SessionStore

MIN_DIM = 16
MAX_DIM = 4096
MAX_MODEL_KEY_LEN = 128
# Centroids use each person's most recent embeddings only.
MAX_CENTROID_EMBEDDINGS = 200
# Embedding ``source`` per modality; model keys are opaque, so the source says which is which.
ENROLL_SOURCES = {"voice": "live:enroll:voice", "face": "live:enroll:face"}
# Pairs scoring below this on every model are not worth a reviewer's attention.
DUPLICATE_SCORE_FLOOR = 0.25
MAX_DUPLICATES = 20


def normalize(vector: np.ndarray) -> np.ndarray | None:
    """L2-normalised ``float32`` copy of ``vector``, or None when it is unusable."""
    vector = np.asarray(vector, dtype="<f4").reshape(-1)
    if not MIN_DIM <= vector.size <= MAX_DIM or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return (vector / norm).astype("<f4")


def decode_vector(encoded: object) -> np.ndarray | None:
    """Decode a base64 ``float32`` vector from the wire; None when malformed."""
    if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_DIM * 6:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw or len(raw) % 4:
        return None
    return normalize(np.frombuffer(raw, dtype="<f4"))


def encode_vector(vector: np.ndarray) -> str:
    """Base64 wire form of a ``float32`` vector."""
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def valid_model_key(model: object) -> bool:
    """Whether ``model`` is a usable model key."""
    return isinstance(model, str) and 0 < len(model.strip()) <= MAX_MODEL_KEY_LEN


def add_embeddings(
    store: SessionStore,
    person_id: str,
    model: str,
    vectors: list[np.ndarray],
    *,
    source: str,
    session_id: str | None = None,
) -> int:
    """Store ``vectors`` for ``person_id`` under ``model``; returns how many were kept."""
    now = time.time()
    rows = []
    for vector in vectors:
        normalized = normalize(vector)
        if normalized is None:
            continue
        rows.append(
            {
                "person_id": person_id,
                "model": model,
                "dim": int(normalized.size),
                "vector": normalized.tobytes(),
                "source": source,
                "session_id": session_id,
                "created_at": now,
            }
        )
    if rows:
        with store.engine.begin() as conn:
            conn.execute(insert(schema.voice_embeddings), rows)
    return len(rows)


def delete_embeddings(store: SessionStore, person_id: str, *, model: str | None = None) -> int:
    """Forget a person's voice (all models, or one); returns the number of vectors removed."""
    ve = schema.voice_embeddings
    stmt = delete(ve).where(ve.c.person_id == person_id)
    if model:
        stmt = stmt.where(ve.c.model == model)
    with store.engine.begin() as conn:
        return conn.execute(stmt).rowcount or 0


def centroids(store: SessionStore, model: str, *, person_id: str | None = None) -> dict[str, dict[str, Any]]:
    """``{person_id: {"centroid": ndarray, "count": int}}`` for ``model``.

    Vectors whose dimension differs from the most common one for the model are
    ignored (a client that reused a model key for another model).
    """
    ve = schema.voice_embeddings
    stmt = select(ve.c.person_id, ve.c.dim, ve.c.vector).where(ve.c.model == model).order_by(ve.c.id.desc())
    if person_id:
        stmt = stmt.where(ve.c.person_id == person_id)
    with store.engine.connect() as conn:
        rows = conn.execute(stmt).all()
    if not rows:
        return {}
    dim = Counter(row.dim for row in rows).most_common(1)[0][0]
    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if row.dim == dim and len(row.vector) == dim * 4 and len(vectors[row.person_id]) < MAX_CENTROID_EMBEDDINGS:
            vectors[row.person_id].append(np.frombuffer(row.vector, dtype="<f4"))
    out: dict[str, dict[str, Any]] = {}
    for pid, person_vectors in vectors.items():
        centroid = normalize(np.mean(person_vectors, axis=0)) if person_vectors else None
        if centroid is not None:
            out[pid] = {"centroid": centroid, "count": len(person_vectors)}
    return out


def gallery(store: SessionStore, model: str) -> list[dict[str, Any]]:
    """Gallery entries for ``model``: non-archived people with at least one embedding."""
    by_person = centroids(store, model)
    return [
        {
            "person_id": person["id"],
            "name": person["name"],
            "centroid": encode_vector(by_person[person["id"]]["centroid"]),
            "count": by_person[person["id"]]["count"],
        }
        for person in memories.list_people(store)
        if person["id"] in by_person
    ]


def modality(source: str) -> str | None:
    """``voice`` or ``face`` for an embedding source, None when the source does not say."""
    return next((name for name, enroll in ENROLL_SOURCES.items() if source == enroll), None)


def identity_summary(store: SessionStore, person_id: str) -> list[dict[str, Any]]:
    """Per model key: modality, sample count, enrollment dates and source sessions for one person."""
    ve = schema.voice_embeddings
    stmt = (
        select(ve.c.model, ve.c.source, ve.c.session_id, ve.c.created_at)
        .where(ve.c.person_id == person_id)
        .order_by(ve.c.created_at)
    )
    with store.engine.connect() as conn:
        rows = conn.execute(stmt).all()
    models: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = models.setdefault(
            row.model,
            {"model": row.model, "modality": None, "count": 0, "first_at": row.created_at, "sessions": []},
        )
        entry["modality"] = entry["modality"] or modality(row.source)
        entry["count"] += 1
        entry["last_at"] = row.created_at
        if row.session_id and row.session_id not in entry["sessions"]:
            entry["sessions"].append(row.session_id)
    return list(models.values())


def identity_counts(store: SessionStore) -> dict[str, dict[str, int]]:
    """``{person_id: {modality: samples}}``; samples whose modality is unknown count as ``other``."""
    ve = schema.voice_embeddings
    with store.engine.connect() as conn:
        rows = conn.execute(
            select(ve.c.person_id, ve.c.source, func.count()).group_by(ve.c.person_id, ve.c.source)
        ).all()
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pid, source, count in rows:
        counts[pid][modality(source) or "other"] += count
    return {pid: dict(by_modality) for pid, by_modality in counts.items()}


def duplicate_candidates(store: SessionStore, *, person_id: str | None = None) -> list[dict[str, Any]]:
    """Pairs of people who may be the same person, most similar first.

    Each pair carries the cosine similarity of their centroids for every model
    key both have samples for, and whether their names match. With
    ``person_id``, only pairs involving that person.
    """
    people = {p["id"]: p for p in memories.list_people(store, include_archived=True)}
    ve = schema.voice_embeddings
    with store.engine.connect() as conn:
        model_rows = conn.execute(select(ve.c.model, ve.c.source).distinct()).all()
    model_modality: dict[str, str | None] = {}
    for model, source in model_rows:
        model_modality[model] = model_modality.get(model) or modality(source)

    pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def pair(a: str, b: str) -> dict[str, Any]:
        key = (a, b) if a < b else (b, a)
        return pairs.setdefault(key, {"people": [people[key[0]], people[key[1]]], "scores": [], "same_name": False})

    for model, kind in model_modality.items():
        by_person = {pid: c["centroid"] for pid, c in centroids(store, model).items() if pid in people}
        ids = sorted(by_person)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                score = float(by_person[a] @ by_person[b])
                if score >= DUPLICATE_SCORE_FLOOR:
                    pair(a, b)["scores"].append({"model": model, "modality": kind, "score": round(score, 4)})
    by_name: dict[str, list[str]] = defaultdict(list)
    for pid, p in people.items():
        by_name[p["name"].strip().casefold()].append(pid)
    for ids in by_name.values():
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                pair(a, b)["same_name"] = True

    out = [p for (a, b), p in pairs.items() if person_id is None or person_id in (a, b)]
    out.sort(key=lambda p: (max((s["score"] for s in p["scores"]), default=0.0), p["same_name"]), reverse=True)
    return out[:MAX_DUPLICATES]


def create_voice_id_router() -> APIRouter:
    """Build the ``/api/voice-id`` router (``503`` while the people store is unavailable)."""
    router = APIRouter(prefix="/api/voice-id", tags=["voice-id"])

    def _store() -> SessionStore:
        try:
            store = memories.live_store()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"people store unavailable: {exc}") from None
        if store is None:
            raise HTTPException(status_code=503, detail="people store unavailable (MONITORING_ENABLED is off)")
        return store

    def _gallery(model: str) -> dict[str, Any]:
        return {"model": model, "people": gallery(_store(), model)}

    def _delete(person_id: str, model: str | None) -> dict[str, Any]:
        return {"person_id": person_id, "deleted": delete_embeddings(_store(), person_id, model=model)}

    @router.get("/gallery")
    async def get_gallery(model: str = Query(min_length=1, max_length=MAX_MODEL_KEY_LEN)):
        return await asyncio.to_thread(_gallery, model.strip())

    @router.delete("/people/{person_id}/embeddings")
    async def forget_voice(person_id: str, model: str | None = Query(default=None, max_length=MAX_MODEL_KEY_LEN)):
        return await asyncio.to_thread(_delete, person_id, model)

    return router

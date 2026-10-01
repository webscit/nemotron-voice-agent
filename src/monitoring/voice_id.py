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
from sqlalchemy import delete, insert, select

from monitoring import memories, schema
from monitoring.store import SessionStore

MIN_DIM = 16
MAX_DIM = 4096
MAX_MODEL_KEY_LEN = 128
# Centroids use each person's most recent embeddings only.
MAX_CENTROID_EMBEDDINGS = 200


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

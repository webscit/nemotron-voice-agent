# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review API: browse recorded sessions and run human review activities.

Mounted by ``server.py`` under ``/api/review`` when ``MONITORING_ENABLED`` is
true. Everything here is CPU-only (DB reads, file serving, WER rescoring):
GPU work is only *queued* for the dreamer, which still waits for idle.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from monitoring import schema
from monitoring.config import MonitoringConfig
from monitoring.jobs import JOB_REGISTRY
from monitoring.jobs.reasr import live_source_name, score_session
from monitoring.jobs.runner import load_dreamer_config
from monitoring.jobs.wer import error_rates, normalize
from monitoring.store import ArtifactStore, SessionStore

_ANNOTATOR_RE = re.compile(r"^[\w .@-]{1,64}$")


class ReferenceIn(BaseModel):
    """A human ASR reference for one user turn."""

    session_id: str
    turn_idx: int
    text: str = Field(max_length=4000)


class ReferenceBatchIn(BaseModel):
    """Human references saved by one annotator."""

    annotator: str
    items: list[ReferenceIn] = Field(min_length=1, max_length=500)


class JobsIn(BaseModel):
    """(Re)queue a job for sessions."""

    kind: str
    session_ids: list[str] = Field(min_length=1, max_length=1000)


def _turn_target(session_id: str, turn_idx: int) -> str:
    return f"{session_id}:{turn_idx}"


def _split_target(target: str) -> tuple[str, int]:
    session_id, _, idx = target.rpartition(":")
    return session_id, int(idx)


def _session_summary(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("config") or {}
    return {
        "id": row["id"],
        "example": row["example"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "end_reason": row["end_reason"],
        "language": config.get("language"),
        "models": {
            "asr": (config.get("asr") or {}).get("model"),
            "llm": (config.get("llm") or {}).get("model"),
            "tts": (config.get("tts") or {}).get("model"),
            "voice": (config.get("tts") or {}).get("voice"),
        },
    }


class ReviewService:
    """Queries and actions behind the review endpoints (synchronous, thread-safe)."""

    def __init__(self, store: SessionStore, artifacts: ArtifactStore, dreamer_config: dict[str, Any]):
        """Bind to the stores; ``dreamer_config`` names the reasr reference/candidates."""
        self.store = store
        self.artifacts = artifacts
        self.dreamer_config = dreamer_config

    # ------------------------------------------------------------- helpers
    @property
    def _reference_name(self) -> str | None:
        reference = (self.dreamer_config.get("reasr") or {}).get("reference")
        return reference.get("name") if reference else None

    @property
    def _candidate_names(self) -> list[str]:
        return [c["name"] for c in (self.dreamer_config.get("reasr") or {}).get("candidates") or []]

    def _rows(self, stmt) -> list[dict[str, Any]]:
        with self.store.engine.connect() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings()]

    # ------------------------------------------------------------ sessions
    def list_sessions(self, *, limit: int, offset: int) -> dict[str, Any]:
        """Most recent sessions with turn counts, review progress, WER and job states."""
        s, t, m, a, j = schema.sessions, schema.turns, schema.media, schema.annotations, schema.jobs
        sessions = self._rows(select(s).order_by(s.c.started_at.desc()).limit(limit).offset(offset))
        ids = [row["id"] for row in sessions]
        with self.store.engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(s)).scalar_one()
        if not ids:
            return {"total": total, "sessions": []}
        user_turns = dict(
            self._rows_pairs(
                select(t.c.session_id, func.count())
                .where(t.c.session_id.in_(ids), t.c.user_text.is_not(None), t.c.user_text != "")
                .group_by(t.c.session_id)
            )
        )
        audio_turns = dict(
            self._rows_pairs(
                select(m.c.session_id, func.count(func.distinct(m.c.turn_idx)))
                .where(m.c.session_id.in_(ids), m.c.modality == "audio_user")
                .group_by(m.c.session_id)
            )
        )
        reviewed = dict(
            self._rows_pairs(
                select(a.c.session_id, func.count(func.distinct(a.c.target_id)))
                .where(a.c.session_id.in_(ids), a.c.kind == "transcript", a.c.source.like("human:%"))
                .group_by(a.c.session_id)
            )
        )
        wer: dict[str, dict[str, float | None]] = defaultdict(dict)
        for row in self._rows(
            select(a.c.session_id, a.c.source, a.c.value).where(a.c.session_id.in_(ids), a.c.kind == "wer_summary")
        ):
            wer[row["session_id"]][row["source"]] = (row["value"] or {}).get("wer")
        jobs: dict[str, dict[str, str]] = defaultdict(dict)
        for row in self._rows(select(j.c.target, j.c.kind, j.c.status).where(j.c.target.in_(ids))):
            jobs[row["target"]][row["kind"]] = row["status"]
        return {
            "total": total,
            "sessions": [
                {
                    **_session_summary(row),
                    "user_turns": user_turns.get(row["id"], 0),
                    "audio_turns": audio_turns.get(row["id"], 0),
                    "reviewed_turns": reviewed.get(row["id"], 0),
                    "wer": wer.get(row["id"], {}),
                    "jobs": jobs.get(row["id"], {}),
                }
                for row in sessions
            ],
        }

    def _rows_pairs(self, stmt) -> list[tuple[Any, Any]]:
        with self.store.engine.connect() as conn:
            return [tuple(r) for r in conn.execute(stmt)]

    def session_detail(self, session_id: str) -> dict[str, Any]:
        """Everything the session explorer shows, grouped per turn."""
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        live_source = live_source_name(session.get("config") or {})
        turns = {
            row["idx"]: {**row, "audio": {"user": [], "bot": []}, "images": [], "llm_calls": []}
            for row in self.store.rows("turns", session_id)
        }

        def turn(idx: int | None) -> dict[str, Any] | None:
            if idx is None:
                return None
            return turns.setdefault(idx, {"idx": idx, "audio": {"user": [], "bot": []}, "images": [], "llm_calls": []})

        conversation_audio = None
        for media in self.store.rows("media", session_id):
            entry = {
                "id": media["id"],
                "key": media["artifact_key"],
                "mime": media["mime"],
                "duration_secs": media["duration_secs"],
                "source": media["source"],
                "ts": media["ts"],
            }
            if media["modality"] == "audio_conversation":
                conversation_audio = entry
            elif media["modality"] in ("audio_user", "audio_bot"):
                target = turn(media["turn_idx"])
                if target is not None:
                    target["audio"][media["modality"].removeprefix("audio_")].append(entry)
            elif media["modality"] in ("image", "video_keyframe"):
                target = turn(media["turn_idx"])
                if target is not None:
                    target["images"].append({**entry, "width": media["width"], "height": media["height"]})

        for call in self.store.rows("llm_calls", session_id):
            target = turn(call["turn_idx"])
            if target is not None:
                target["llm_calls"].append(
                    {
                        key: call[key]
                        for key in (
                            "id",
                            "started_at",
                            "model",
                            "ttfb",
                            "prompt_tokens",
                            "completion_tokens",
                            "n_images",
                            "interrupted",
                            "output_text",
                            "function_calls",
                        )
                    }
                )

        latency: dict[int, float] = {}
        for metric in self.store.rows("metrics", session_id, name="user_bot_latency"):
            if metric["turn_idx"] is not None:
                latency.setdefault(metric["turn_idx"], metric["value"])

        transcripts = self._transcripts_by_turn(session_id)
        wer_by_turn: dict[int, dict[str, Any]] = defaultdict(dict)
        wer_summary = {}
        for annotation in self.store.annotations_for(session_id):
            if annotation["kind"] == "wer":
                wer_by_turn[_split_target(annotation["target_id"])[1]][annotation["source"]] = annotation["value"]
            elif annotation["kind"] == "wer_summary":
                wer_summary[annotation["source"]] = annotation["value"]

        out_turns = []
        for idx in sorted(turns):
            row = turns[idx]
            by_source = transcripts.get(idx, {})
            if row.get("user_text") and row["audio"]["user"]:
                by_source = {live_source: {"text": row["user_text"], "at": 0.0}, **by_source}
            human = self._latest_human(by_source)
            out_turns.append(
                {
                    **row,
                    "transcripts": {k: v["text"] for k, v in by_source.items()},
                    "human_reference": human,
                    "wer": wer_by_turn.get(idx, {}),
                    "user_bot_latency": latency.get(idx),
                }
            )
        jobs = [job for job in self.store.jobs() if job["target"] == session_id]
        return {
            "session": {**_session_summary(session), "config": session.get("config")},
            "live_source": live_source,
            "reference_source": self._reference_name,
            "conversation_audio": conversation_audio,
            "turns": out_turns,
            "wer_summary": wer_summary,
            "jobs": [{k: job[k] for k in ("kind", "status", "error", "attempts", "finished_at")} for job in jobs],
        }

    def llm_call(self, session_id: str, call_id: int) -> dict[str, Any]:
        """Full stored LLM input/output of one call."""
        calls = self.store.rows("llm_calls", session_id, id=call_id)
        if not calls:
            raise KeyError(call_id)
        return calls[0]

    # -------------------------------------------------------------- ASR queue
    def _transcripts_by_turn(self, session_id: str) -> dict[int, dict[str, dict[str, Any]]]:
        """``{turn_idx: {source: {"text", "at"}}}`` keeping the latest per source."""
        out: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in self.store.annotations_for(session_id, kind="transcript"):
            _, idx = _split_target(row["target_id"])
            out[idx][row["source"]] = {"text": (row["value"] or {}).get("text", ""), "at": row["created_at"]}
        return out

    @staticmethod
    def _latest_human(by_source: dict[str, dict[str, Any] | str]) -> dict[str, Any] | None:
        humans = [
            (value["at"], source, value["text"])
            for source, value in by_source.items()
            if source.startswith("human:") and isinstance(value, dict)
        ]
        if not humans:
            return None
        at, source, text = max(humans)
        return {"source": source, "text": text, "at": at}

    def asr_queue(
        self, *, status: str, limit: int, session_id: str | None, sort: str = "recent", offset: int = 0
    ) -> dict[str, Any]:
        """User turns with audio, most recent first (or most live-vs-reference disagreement first)."""
        reference = self._reference_name
        session_ids = [session_id] if session_id else self.store.ended_session_ids()
        items = []
        for sid in session_ids:
            session = self.store.get_session(sid)
            if session is None:
                continue
            config = session.get("config") or {}
            live_source = live_source_name(config)
            audio: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for media in self.store.rows("media", sid, modality="audio_user"):
                audio[media["turn_idx"]].append({"key": media["artifact_key"], "duration_secs": media["duration_secs"]})
            if not audio:
                continue
            transcripts = self._transcripts_by_turn(sid)
            for turn in self.store.rows("turns", sid):
                idx = turn["idx"]
                if idx not in audio:
                    continue
                by_source = transcripts.get(idx, {})
                human = self._latest_human(by_source)
                models = {src: v["text"] for src, v in by_source.items() if not src.startswith("human:")}
                live_text = turn["user_text"] or ""
                ref_text = models.get(reference) if reference else None
                disagreement = error_rates(ref_text, live_text)["wer"] if ref_text is not None else None
                items.append(
                    {
                        "session_id": sid,
                        "turn_idx": idx,
                        "started_at": turn["user_started_at"] or session["started_at"],
                        "language": turn["language"] or config.get("language"),
                        "audio": audio[idx],
                        "live": {"source": live_source, "text": live_text},
                        "models": models,
                        "reference_source": reference,
                        "human_reference": human,
                        "disagreement": disagreement,
                        "agree": ref_text is not None and normalize(ref_text) == normalize(live_text),
                    }
                )
        # Counts cover every turn; ``status`` only filters the returned items.
        counts = {
            "total": len(items),
            "open": sum(1 for i in items if not i["human_reference"]),
            "agreeing": sum(1 for i in items if i["agree"] and not i["human_reference"]),
        }
        if status == "open":
            items = [i for i in items if not i["human_reference"]]
        elif status == "reviewed":
            items = [i for i in items if i["human_reference"]]
        if sort == "disagreement":
            # Known disagreement first (largest first), then turns still waiting for the reference model.
            items.sort(key=lambda i: (i["disagreement"] is None, -(i["disagreement"] or 0.0), -i["started_at"]))
        else:
            items.sort(key=lambda i: (-i["started_at"], -i["turn_idx"]))
        return {**counts, "matching": len(items), "items": items[offset : offset + limit]}

    def save_references(self, annotator: str, items: list[ReferenceIn]) -> dict[str, Any]:
        """Store human references and rescore the affected sessions inline."""
        rows = [
            {
                "session_id": item.session_id,
                "target_type": "turn",
                "target_id": _turn_target(item.session_id, item.turn_idx),
                "source": f"human:{annotator}",
                "source_version": None,
                "kind": "transcript",
                "value": {"text": item.text.strip()},
            }
            for item in items
        ]
        self.store.add_annotations(rows)
        summaries = {}
        for sid in sorted({item.session_id for item in items}):
            summaries[sid] = {
                s["source"]: s["value"]
                for s in score_session(
                    self.store, sid, reference_name=self._reference_name, candidate_names=self._candidate_names
                )
            }
        return {"saved": len(rows), "wer_summary": summaries}

    # ------------------------------------------------------------- dreamer
    def dreamer_status(self) -> dict[str, Any]:
        """Idle/live state, job counts, running and failed jobs."""
        now = time.time()
        live = self.store.open_session_count(now=now, stale_after_secs=self.dreamer_config["live_stale_secs"])
        last = self.store.last_session_activity()
        jobs = self.store.jobs()
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for job in jobs:
            counts[job["kind"]][job["status"]] += 1

        def brief(job):
            return {k: job[k] for k in ("id", "kind", "target", "status", "attempts", "error", "progress")}

        return {
            "live_sessions": live,
            "idle": live == 0 and (last is None or now - last >= self.dreamer_config["idle_grace_secs"]),
            "last_activity": last,
            "counts": {kind: dict(v) for kind, v in counts.items()},
            "running": [brief(j) for j in jobs if j["status"] == "running"],
            "failed": [brief(j) for j in jobs if j["status"] == "failed" or (j["error"] and j["status"] != "done")],
            "pending": sum(1 for j in jobs if j["status"] == "pending"),
        }

    def requeue(self, kind: str, session_ids: list[str]) -> int:
        """Queue ``kind`` from scratch for sessions (the dreamer still waits for idle)."""
        for sid in session_ids:
            self.store.reset_job(kind, sid)
        return len(session_ids)

    def activities(self) -> list[dict[str, Any]]:
        """Review activities available, with their open item counts."""
        queue = self.asr_queue(status="open", limit=0, session_id=None)
        return [
            {
                "id": "asr",
                "title": "ASR reference",
                "job": "reasr",
                "description": "Set the human reference transcript of user turns; rescored instantly.",
                "open": queue["open"],
            }
        ]


def create_review_router(
    settings: MonitoringConfig,
    store: SessionStore,
    artifacts: ArtifactStore,
    dreamer_config: dict[str, Any] | None = None,
) -> APIRouter:
    """Build the ``/api/review`` router."""
    service = ReviewService(store, artifacts, dreamer_config or load_dreamer_config())
    router = APIRouter(prefix="/api/review", tags=["review"])

    async def run(fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    def check_annotator(name: str) -> str:
        name = name.strip()
        if not _ANNOTATOR_RE.match(name):
            raise HTTPException(status_code=422, detail="annotator: 1-64 letters, digits, space, . @ - _")
        return name

    @router.get("/activities")
    async def activities():
        return {"activities": await run(service.activities)}

    @router.get("/sessions")
    async def sessions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        return await run(service.list_sessions, limit=limit, offset=offset)

    @router.get("/sessions/{session_id}")
    async def session(session_id: str):
        try:
            return await run(service.session_detail, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found") from None

    @router.get("/sessions/{session_id}/llm-calls/{call_id}")
    async def llm_call(session_id: str, call_id: int):
        try:
            return await run(service.llm_call, session_id, call_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="llm call not found") from None

    @router.get("/artifacts/{key:path}")
    async def artifact(key: str):
        read_path = getattr(artifacts, "read_path", None)
        try:
            if read_path is not None:
                path = read_path(key)
                if not path.is_file():
                    raise FileNotFoundError(key)
                return FileResponse(path)  # honors HTTP Range (audio seeking)
            data = await run(artifacts.get, key)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="artifact not found") from None
        return Response(data)

    @router.get("/asr/queue")
    async def asr_queue(
        status: str = Query("open", pattern="^(open|reviewed|all)$"),
        limit: int = Query(50, ge=0, le=1000),
        offset: int = Query(0, ge=0),
        sort: str = Query("recent", pattern="^(recent|disagreement)$"),
        session_id: str | None = None,
    ):
        return await run(service.asr_queue, status=status, limit=limit, offset=offset, sort=sort, session_id=session_id)

    @router.post("/asr/references")
    async def save_references(payload: ReferenceBatchIn):
        annotator = check_annotator(payload.annotator)
        return await run(service.save_references, annotator, payload.items)

    @router.get("/dreamer")
    async def dreamer():
        return await run(service.dreamer_status)

    @router.post("/jobs")
    async def jobs(payload: JobsIn):
        if payload.kind not in JOB_REGISTRY:
            raise HTTPException(status_code=422, detail=f"unknown job kind {payload.kind!r}")
        return {"queued": await run(service.requeue, payload.kind, payload.session_ids)}

    return router

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relational schema of the session store (SQLAlchemy Core, dialect-neutral).

All timestamps are Unix epoch seconds (float). Binary payloads (audio, images,
video) never live in the database: ``media.artifact_key`` points into the
artifact store.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

SCHEMA_VERSION = 3

metadata = MetaData()

schema_version = Table(
    "schema_version",
    metadata,
    Column("version", Integer, primary_key=True),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("example", String(128)),
    Column("started_at", Float, nullable=False),
    Column("ended_at", Float),
    Column("last_seen_at", Float),
    Column("end_reason", String(64)),
    # Snapshot of everything that defines the pipeline variant (models, recipe,
    # language, voice, turn detection, git sha, client tools): the A/B key.
    Column("config", JSON),
    Column("artifact_prefix", String(256)),
)

turns = Table(
    "turns",
    metadata,
    Column("session_id", String(64), primary_key=True),
    Column("idx", Integer, primary_key=True),
    Column("user_text", Text),
    Column("user_started_at", Float),
    Column("user_stopped_at", Float),
    Column("bot_text", Text),
    Column("bot_started_at", Float),
    Column("bot_stopped_at", Float),
    Column("interrupted", Boolean, default=False),
    Column("language", String(32)),
)

events = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("ts", Float, nullable=False),
    Column("kind", String(64), nullable=False),
    Column("processor", String(128)),
    Column("data", JSON),
    Index("ix_events_session", "session_id", "ts"),
)

metrics = Table(
    "metrics",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("ts", Float, nullable=False),
    Column("processor", String(128)),
    Column("model", String(256)),
    Column("name", String(128), nullable=False),
    Column("value", Float),
    Index("ix_metrics_session", "session_id", "name"),
)

media = Table(
    "media",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("ts", Float, nullable=False),
    # audio_user | audio_bot | audio_conversation | image | video_keyframe | video_chunk
    Column("modality", String(32), nullable=False),
    # camera | context | tool:<name> | pipeline | ...
    Column("source", String(128)),
    # Free-form link to what produced it (tool_call_id, llm call id, ...).
    Column("ref", String(128)),
    Column("sha256", String(64), nullable=False),
    Column("mime", String(64)),
    Column("width", Integer),
    Column("height", Integer),
    Column("sample_rate", Integer),
    Column("duration_secs", Float),
    Column("artifact_key", String(512), nullable=False),
    UniqueConstraint("session_id", "modality", "sha256", name="uq_media_content"),
    Index("ix_media_session", "session_id", "modality"),
)

llm_calls = Table(
    "llm_calls",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("started_at", Float, nullable=False),
    Column("ended_at", Float),
    Column("processor", String(128)),
    Column("model", String(256)),
    # Full LLM input; inline images are replaced by {"type": "media_ref", "sha256": ...}.
    Column("messages", JSON),
    Column("tools", JSON),
    Column("output_text", Text),
    Column("function_calls", JSON),
    Column("ttfb", Float),
    Column("prompt_tokens", Integer),
    Column("completion_tokens", Integer),
    Column("n_images", Integer, default=0),
    Column("image_pixels", Integer, default=0),
    Column("interrupted", Boolean, default=False),
    Index("ix_llm_calls_session", "session_id", "turn_idx"),
)

annotations = Table(
    "annotations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    # session | turn | media | llm_call
    Column("target_type", String(32), nullable=False),
    Column("target_id", String(128), nullable=False),
    # Model name for automatic annotations, ``human:<name>`` for human labels.
    Column("source", String(256), nullable=False),
    Column("source_version", String(128)),
    # transcript | wer | safety | memory | ...
    Column("kind", String(64), nullable=False),
    Column("value", JSON),
    Column("created_at", Float, nullable=False),
    Index("ix_annotations_target", "target_type", "target_id", "kind"),
    Index("ix_annotations_session", "session_id", "kind"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kind", String(64), nullable=False),
    Column("target", String(128), nullable=False),
    # pending | running | done | failed | cancelled
    Column("status", String(16), nullable=False, default="pending"),
    Column("params", JSON),
    Column("progress", JSON),
    Column("attempts", Integer, nullable=False, default=0),
    Column("error", Text),
    Column("created_at", Float, nullable=False),
    Column("started_at", Float),
    Column("finished_at", Float),
    UniqueConstraint("kind", "target", name="uq_jobs_kind_target"),
    Index("ix_jobs_status", "status", "created_at"),
)

# Small shared state between processes (dreamer status heartbeat, pause flag).
kv = Table(
    "kv",
    metadata,
    Column("key", String(128), primary_key=True),
    Column("value", JSON),
    Column("updated_at", Float, nullable=False),
)


# ----------------------------------------------------------------- people
# People the agent talks to. Until voice identification exists, a session is
# attributed to the person picked in the live client, and review can reassign
# whole sessions or single turns (the per-turn rows are voice-ID enrollment data).
people = Table(
    "people",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("name", String(128), nullable=False),
    Column("created_at", Float, nullable=False),
    Column("archived", Boolean, nullable=False, default=False),
)

speaker_assignments = Table(
    "speaker_assignments",
    metadata,
    Column("session_id", String(64), primary_key=True),
    # -1 = the whole session; a turn row overrides it for that turn.
    Column("turn_idx", Integer, primary_key=True),
    Column("person_id", String(32), nullable=False),
    # ``live:picker`` or ``human:<name>``.
    Column("source", String(256), nullable=False),
    Column("created_at", Float, nullable=False),
    Index("ix_speaker_person", "person_id"),
)

# Facts about a person extracted by the ``dream`` job (or written by a reviewer).
# Rows are never deleted: forgetting and superseding are status changes.
memories = Table(
    "memories",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("person_id", String(32), nullable=False),
    Column("text", Text, nullable=False),
    # identity | preference | relationship | routine | event | health | other
    Column("category", String(32)),
    Column("language", String(32)),
    Column("confidence", Float),
    # proposed (waiting for review) | active (used live) | forgotten | superseded
    Column("status", String(16), nullable=False),
    # Model name for extracted memories, ``human:<name>`` for corrections.
    Column("source", String(256), nullable=False),
    Column("source_version", String(256)),
    # Memory this one replaces once approved or used.
    Column("supersedes", Integer),
    Column("superseded_by", Integer),
    Column("reviewed_by", String(256)),
    Column("reviewed_at", Float),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Index("ix_memories_person", "person_id", "status"),
)

memory_evidence = Table(
    "memory_evidence",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("memory_id", Integer, nullable=False),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("quote", Text),
    Index("ix_memory_evidence_memory", "memory_id"),
    Index("ix_memory_evidence_session", "session_id"),
)

# Memories injected into a live session prompt.
memory_uses = Table(
    "memory_uses",
    metadata,
    Column("memory_id", Integer, primary_key=True),
    Column("session_id", String(64), primary_key=True),
    Column("ts", Float, nullable=False),
    Index("ix_memory_uses_session", "session_id"),
)

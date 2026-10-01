# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live side of memories: who is talking and what the agent remembers about them.

The client sends ``person_id`` (the "Who's talking" picker). At session start the
pipeline loads that person's ``active`` memories into the pinned system prompt,
then records the attribution and which memories were used. Every step is best
effort: a missing monitoring store or a database error never blocks a session.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

PERSON_MEMORY_ADDON_KEY = "person_memory_addon"
MAX_PROMPT_MEMORIES = 30


@dataclass
class PersonContext:
    """The person talking and the memories injected into the prompt."""

    person: dict[str, Any]
    memories: list[dict[str, Any]] = field(default_factory=list)

    @property
    def snapshot(self) -> dict[str, Any]:
        """Session-config entry (``config.person``)."""
        return {"id": self.person["id"], "name": self.person["name"]}

    def prompt_replacements(self) -> dict[str, str]:
        """Placeholders of the ``person_memory_addon`` prompt block."""
        from monitoring.memories import format_memories

        return {"person_name": self.person["name"], "memories": format_memories(self.memories)}


def _load(person_id: str) -> PersonContext | None:
    from monitoring import memories

    store = memories.live_store()
    if store is None:
        return None
    person = memories.get_person(store, person_id)
    if person is None or person.get("archived"):
        return None
    return PersonContext(person, memories.active_memories(store, person_id, limit=MAX_PROMPT_MEMORIES))


async def load_person_context(person_id: object) -> PersonContext | None:
    """Person and active memories for ``person_id``, or None (unknown, archived, monitoring off, error)."""
    if not isinstance(person_id, str) or not person_id.strip():
        return None
    try:
        return await asyncio.to_thread(_load, person_id.strip())
    except Exception as exc:
        logger.opt(exception=exc).warning("Could not load person memories; continuing without them")
        return None


def _record(session_id: str, context: PersonContext) -> None:
    from monitoring import memories

    store = memories.live_store()
    if store is None:
        return
    memories.assign_speaker(store, session_id, context.person["id"], source="live:picker")
    memories.record_uses(store, session_id, [m["id"] for m in context.memories])


async def record_person_session(session_id: str | None, context: PersonContext | None) -> None:
    """Attribute the session to the person and log the memories used (best effort)."""
    if not session_id or context is None:
        return
    try:
        await asyncio.to_thread(_record, session_id, context)
    except Exception as exc:
        logger.opt(exception=exc).warning("Could not record the session person / memory uses")

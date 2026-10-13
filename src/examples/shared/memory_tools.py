# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side ``remember`` / ``forget`` tools: the person edits their own memories live.

The tools act on the person whose memories are in the pinned prompt, and only
while that person is known for sure (picked in the client, or recognised with a
trusted voice ID tier). ``remember`` stores an ``active`` but unreviewed memory
(origin ``conversation``, source ``live:<session_id>``) with the user's words as
evidence; ``forget`` marks the best matching memory ``forgotten``. Both refresh
the pinned prompt, so the LLM sees the change on its next reply.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from examples.shared.person_memory import PersonContext

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.llm_service import FunctionCallParams

REMEMBER_TOOL_NAME = "remember"
FORGET_TOOL_NAME = "forget"
MEMORY_TOOL_NAMES = (REMEMBER_TOOL_NAME, FORGET_TOOL_NAME)
MAX_FACT_LEN = 300
MAX_QUOTE_LEN = 500
# Share of the query words a memory must contain to be forgotten (1.0 when it contains the whole query).
FORGET_MIN_SCORE = 0.5
MAX_CANDIDATES = 5
# Leading speaker / presence tags of a user message ("[speaker: Alice] ...").
_TAG_RE = re.compile(r"^(\[[^\]]*\]\s*)+")

UNKNOWN_PERSON_ERROR = (
    "You do not know for sure who is talking, so nothing was changed. If they are new, ask for their name and "
    "offer to remember them first; if you are unsure, ask them to confirm who they are."
)


def remember_tool_schema() -> FunctionSchema:
    """LLM schema of the server-side ``remember`` tool."""
    return FunctionSchema(
        name=REMEMBER_TOOL_NAME,
        description=(
            "Save one short, lasting fact about the person you are talking with, so you remember it in later "
            "conversations. Call it when they ask you to remember something, or share a stable fact about "
            "themselves (preferences, important people, routines, plans). One fact per call. Never save "
            "passwords, addresses, payment details or anything they did not want kept. Do not announce that "
            "you are saving it; acknowledge naturally."
        ),
        properties={
            "fact": {
                "type": "string",
                "description": (
                    "One short third-person sentence naming the person, in the language of the conversation "
                    '(e.g. "Alice has a dog named Mochi.").'
                ),
            }
        },
        required=["fact"],
    )


def forget_tool_schema() -> FunctionSchema:
    """LLM schema of the server-side ``forget`` tool."""
    return FunctionSchema(
        name=FORGET_TOOL_NAME,
        description=(
            "Forget something you remember about the person you are talking with, when they ask you to forget "
            "it or tell you it is no longer true. If other memories also match, ask them whether those should "
            "be forgotten too."
        ),
        properties={
            "query": {
                "type": "string",
                "description": 'A few words of the memory to forget, e.g. "dog" or "works at the bakery".',
            }
        },
        required=["query"],
    )


def with_memory_tools(tools: object) -> ToolsSchema:
    """``tools`` (a ``ToolsSchema`` or not given) plus the ``remember`` and ``forget`` tools."""
    schemas = [remember_tool_schema(), forget_tool_schema()]
    if isinstance(tools, ToolsSchema):
        standard = [t for t in tools.standard_tools if getattr(t, "name", None) not in MEMORY_TOOL_NAMES]
        return ToolsSchema(standard_tools=[*standard, *schemas], custom_tools=tools.custom_tools)
    return ToolsSchema(standard_tools=schemas)


def latest_user_text(context: LLMContext | None) -> str:
    """Text of the latest user message, without speaker / presence tags ("" when there is none)."""
    if context is None:
        return ""
    for message in reversed(context.get_messages()):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("text"))
        return _TAG_RE.sub("", str(content or "")).strip()[:MAX_QUOTE_LEN]
    return ""


def _source(session_id: str | None) -> str:
    return f"live:{session_id or 'unrecorded'}"


def _remember(
    person_id: str, fact: str, *, session_id: str | None, turn_idx: int, quote: str, language: str | None
) -> dict[str, Any]:
    from monitoring import memories

    store = memories.live_store()
    if store is None:
        return {"error": "Memory is off on this server; nothing was saved."}
    wanted = fact.casefold()
    if any(m["text"].casefold() == wanted for m in memories.usable_memories(store, [person_id])):
        return {"status": "already remembered", "fact": fact}
    evidence = [{"session_id": session_id, "turn_idx": turn_idx, "quote": quote or None}] if session_id else []
    memories.add_memory(
        store,
        {
            "person_id": person_id,
            "text": fact,
            "category": "other",
            "language": language,
            "confidence": 1.0,
            "status": "active",
            "source": _source(session_id),
        },
        evidence,
    )
    return {"status": "saved", "fact": fact}


def _forget(person_id: str, query: str, *, session_id: str | None) -> dict[str, Any]:
    from monitoring import memories

    store = memories.live_store()
    if store is None:
        return {"error": "Memory is off on this server; nothing was forgotten."}
    matches = memories.match_memories(query, memories.usable_memories(store, [person_id]))
    if not matches or matches[0][0] < FORGET_MIN_SCORE:
        return {
            "error": "No memory clearly matches; nothing was forgotten. Ask what exactly to forget.",
            "candidates": [m["text"] for _, m in matches[:MAX_CANDIDATES]],
        }
    best = matches[0][1]
    memories.forget_memory(store, best["id"], by=_source(session_id))
    others = [m["text"] for score, m in matches[1 : MAX_CANDIDATES + 1] if score >= FORGET_MIN_SCORE]
    result: dict[str, Any] = {"status": "forgotten", "forgotten": best["text"]}
    if others:
        result["other_matches"] = others
    return result


class MemoryTools:
    """``remember`` / ``forget`` handlers for one live session."""

    def __init__(
        self,
        *,
        current_person: Callable[[], PersonContext | None],
        refresh: Callable[[], Awaitable[None]],
        session_id: Callable[[], str | None],
        turn_idx: Callable[[], int],
        language: str | None = None,
    ):
        """Bind the tools to the session.

        Args:
            current_person: The person the tools act on, or None when nobody is known for sure.
            refresh: Reloads that person's memories into the pinned prompt.
            session_id: Returns the recorded session id, or None when not recording.
            turn_idx: Returns the recorder's current turn index (``-1`` when not recording).
            language: The session language, stored with remembered facts.
        """
        self._current_person = current_person
        self._refresh = refresh
        self._session_id = session_id
        self._turn_idx = turn_idx
        self._language = language

    async def remember(self, params: FunctionCallParams) -> None:
        """Server-side ``remember(fact)`` tool handler."""
        fact = " ".join(str(params.arguments.get("fact") or "").split())
        await params.result_callback(await self._remember(fact, latest_user_text(getattr(params, "context", None))))

    async def forget(self, params: FunctionCallParams) -> None:
        """Server-side ``forget(query)`` tool handler."""
        query = " ".join(str(params.arguments.get("query") or "").split())
        await params.result_callback(await self._forget(query))

    async def _remember(self, fact: str, quote: str) -> dict[str, Any]:
        if not fact:
            return {"error": "Give the fact to remember."}
        if len(fact) > MAX_FACT_LEN:
            return {"error": f"The fact is too long; keep it under {MAX_FACT_LEN} characters, one fact per call."}
        person = self._current_person()
        if person is None:
            return {"error": UNKNOWN_PERSON_ERROR}
        try:
            result = await asyncio.to_thread(
                _remember,
                person.person["id"],
                fact,
                session_id=self._session_id(),
                turn_idx=self._turn_idx(),
                quote=quote,
                language=self._language,
            )
        except Exception as exc:
            logger.opt(exception=exc).warning("Could not save a memory")
            return {"error": "The memory could not be saved."}
        if result.get("status") == "saved":
            logger.info(f"Memory: remembered for {person.person['name']}: {fact[:120]}")
            await self._refresh()
        return result

    async def _forget(self, query: str) -> dict[str, Any]:
        if not query:
            return {"error": "Say which memory to forget."}
        person = self._current_person()
        if person is None:
            return {"error": UNKNOWN_PERSON_ERROR}
        try:
            result = await asyncio.to_thread(_forget, person.person["id"], query, session_id=self._session_id())
        except Exception as exc:
            logger.opt(exception=exc).warning("Could not forget a memory")
            return {"error": "The memory could not be forgotten."}
        if result.get("status") == "forgotten":
            logger.info(f"Memory: forgot for {person.person['name']}: {result['forgotten'][:120]}")
            await self._refresh()
        return result

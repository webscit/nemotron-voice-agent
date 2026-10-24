# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sentences and intents learned offline by the ``dream`` job.

Approved suggestions live in the session store (``intent_suggestions``). They
are merged into the sentence data the matcher parses, as if they were written
in a sentence file:

- ``sentence``: one more ``data`` block for an existing intent, custom or stock;
- ``intent``: a new custom intent with its client-tool mapping and response.

Every merged block carries ``metadata: {learned_id: <id>}`` so a match says
which suggestion it came from.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from hassil.expression import Group, ListReference, RuleReference
from hassil.parse_expression import parse_sentence
from loguru import logger

LEARNED_ID_KEY = "learned_id"
# Synced from Home Assistant at session start, not declared in the sentence files.
HOME_ASSISTANT_LISTS = frozenset({"name", "area", "floor"})


@dataclass(frozen=True)
class LearnedIntent:
    """One active suggestion, as the matcher uses it."""

    id: int
    kind: str  # sentence | intent
    intent: str
    sentences: tuple[str, ...]
    slots: dict[str, Any] = field(default_factory=dict)
    # ``intent`` kind: ``{"tool": ..., "arguments": {...}}`` (a ``client_tools.yaml`` entry).
    tool: dict[str, Any] | None = None
    response: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> LearnedIntent:
        """Build from an ``intent_suggestions`` row."""
        return cls(
            id=int(row["id"]),
            kind=str(row["kind"]),
            intent=str(row["intent"]),
            sentences=tuple(str(s) for s in row.get("sentences") or [] if str(s).strip()),
            slots=dict(row.get("slots") or {}),
            tool=dict(row["tool"]) if isinstance(row.get("tool"), dict) else None,
            response=row.get("response"),
        )

    def data_block(self) -> dict[str, Any]:
        """The hassil ``data`` block of these sentences."""
        block: dict[str, Any] = {"sentences": list(self.sentences), "metadata": {LEARNED_ID_KEY: self.id}}
        if self.slots:
            block["slots"] = dict(self.slots)
        return block


def sentence_references(template: str) -> tuple[set[str], set[str], set[str]]:
    """Return the ``({list names}, {expansion rule names}, {slot names})`` a hassil template uses.

    Raises ``ValueError`` when the template does not parse.
    """
    try:
        sentence = parse_sentence(template)
    except Exception as exc:  # hassil raises several parse error types
        raise ValueError(f"does not parse: {exc}") from None
    lists: set[str] = set()
    rules: set[str] = set()
    slots: set[str] = set()
    stack = [sentence.expression]
    while stack:
        item = stack.pop()
        if isinstance(item, ListReference):
            slots.add(item.slot_name)
            if item._inline_range_match is None:  # ``{0..100:level}`` needs no list
                lists.add(item.list_name)
        elif isinstance(item, RuleReference):
            rules.add(item.rule_name)
        elif isinstance(item, Group):
            stack.extend(item.items)
    return lists, rules, slots


def sentence_problems(template: str, data: dict[str, Any], *, extra_lists: Iterable[str] = ()) -> list[str]:
    """Why a template cannot be used with sentence ``data`` (empty when it can).

    Every list and expansion rule must be defined in ``data`` (or be one of
    ``extra_lists``), and no list may be a wildcard: a wildcard matches almost
    any sentence.
    """
    try:
        lists, rules, _ = sentence_references(template)
    except ValueError as exc:
        return [str(exc)]
    declared = data.get("lists") or {}
    known = set(declared) | set(extra_lists)
    problems = [f"unknown list {{{name}}}" for name in sorted(lists - known)]
    problems += [f"unknown rule <{name}>" for name in sorted(rules - set(data.get("expansion_rules") or {}))]
    problems += [
        f"wildcard list {{{name}}}"
        for name in sorted(lists & set(declared))
        if isinstance(declared[name], dict) and declared[name].get("wildcard")
    ]
    return problems


def new_intents(learned: Iterable[LearnedIntent]) -> dict[str, LearnedIntent]:
    """Learned ``intent`` rows with a tool mapping, by intent name."""
    return {item.intent: item for item in learned if item.kind == "intent" and item.tool and item.sentences}


def tool_entries(learned: Iterable[LearnedIntent]) -> dict[str, dict[str, Any]]:
    """``client_tools.yaml`` entries of the learned intents."""
    return {name: copy.deepcopy(item.tool or {}) for name, item in new_intents(learned).items()}


def merge_learned(
    data: dict[str, Any], learned: Sequence[LearnedIntent], *, add_intents: bool
) -> tuple[dict[str, Any], list[int]]:
    """Return a copy of hassil sentence ``data`` with the learned sentences merged in, and the ids used.

    A ``sentence`` row is merged when its intent is in ``data`` (or is a learned
    intent added here). ``add_intents`` adds the learned ``intent`` rows with their
    response template; only the custom sentence set takes them.
    """
    merged = copy.deepcopy(data)
    intents = merged.setdefault("intents", {})
    used: list[int] = []
    extra_lists = HOME_ASSISTANT_LISTS

    def usable(item: LearnedIntent) -> bool:
        problems = [
            p for sentence in item.sentences for p in sentence_problems(sentence, merged, extra_lists=extra_lists)
        ]
        if problems:
            logger.warning(f"Intent engine: learned suggestion #{item.id} ignored: {'; '.join(problems)}")
        return not problems

    if add_intents:
        responses = merged.setdefault("responses", {}).setdefault("intents", {})
        for name, item in new_intents(learned).items():
            if name in intents:
                logger.warning(f"Intent engine: learned intent {name} clashes with a sentence file; ignored")
                continue
            if not usable(item):
                continue
            intents[name] = {"data": [item.data_block()]}
            responses.setdefault(name, {})["default"] = item.response or ""
            used.append(item.id)
    for item in learned:
        if item.kind != "sentence" or not item.sentences or item.intent not in intents:
            continue
        body = intents[item.intent]
        if not isinstance(body, dict) or not usable(item):
            continue
        body["data"] = [*(body.get("data") or []), item.data_block()]
        used.append(item.id)
    return merged, used


def load_learned_intents(language: str) -> list[LearnedIntent]:
    """Active learned sentences and intents for a language; empty when monitoring is off."""
    from monitoring import intents, memories

    store = memories.live_store()
    if store is None:
        return []
    return [LearnedIntent.from_row(row) for row in intents.active_suggestions(store, language)]


async def load_learned_intents_async(language: str) -> list[LearnedIntent]:
    """``load_learned_intents`` off the event loop; never raises (a session starts without them)."""
    try:
        return await asyncio.to_thread(load_learned_intents, language)
    except Exception as exc:
        logger.opt(exception=exc).warning("Intent engine: could not load learned sentences; continuing without them")
        return []

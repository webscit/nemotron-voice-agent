# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sentences and intents learned offline by the ``dream`` job.

Approved suggestions live in the session store (``intent_suggestions``). They
are merged into the sentence data the matcher parses, as if they were written
in a sentence file:

- ``sentence``: one more ``data`` block for an existing intent, custom or stock.
  A stock Home Assistant intent gets one copy per stock block of the same shape
  (target slots and fixed slots), so the learned sentence keeps that block's
  ``requires_context`` (entity domains) and response;
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
# They are also the slots that tell Home Assistant which entities a command targets.
HOME_ASSISTANT_LISTS = frozenset({"name", "area", "floor"})
# Stock block keys a learned copy inherits.
_INHERITED_KEYS = ("requires_context", "excludes_context", "response")


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


def sentence_references(
    template: str, expansion_rules: dict[str, str] | None = None
) -> tuple[set[str], set[str], set[str]]:
    """Return the ``({list names}, {expansion rule names}, {slot names})`` a hassil template uses.

    With ``expansion_rules``, the lists and slots of the rules it uses count too.
    Raises ``ValueError`` when the template does not parse.
    """
    lists: set[str] = set()
    rules: set[str] = set()
    slots: set[str] = set()
    pending = [template]
    while pending:
        try:
            sentence = parse_sentence(pending.pop())
        except Exception as exc:  # hassil raises several parse error types
            raise ValueError(f"does not parse: {exc}") from None
        stack = [sentence.expression]
        while stack:
            item = stack.pop()
            if isinstance(item, ListReference):
                slots.add(item.slot_name)
                if item._inline_range_match is None:  # ``{0..100:level}`` needs no list
                    lists.add(item.list_name)
            elif isinstance(item, RuleReference):
                if expansion_rules and item.rule_name not in rules and item.rule_name in expansion_rules:
                    pending.append(expansion_rules[item.rule_name])
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


def _stock_shape(sentence: str, rules: dict[str, str]) -> tuple[frozenset[str], set[str]] | None:
    """``(target slots, all slots)`` of a template, or None when it does not parse."""
    try:
        _, _, slots = sentence_references(sentence, rules)
    except ValueError:
        return None
    return frozenset(slots & HOME_ASSISTANT_LISTS), slots


def _fits(item: LearnedIntent, sentence: str, block: dict[str, Any], rules: dict[str, str]) -> bool:
    """Whether a learned sentence has the shape of a stock block.

    Same target slots, and every slot the block fixes is fixed to the same value by
    the suggestion or filled by the sentence.
    """
    mine = _stock_shape(sentence, rules)
    theirs = [_stock_shape(s, rules) for s in block.get("sentences") or []]
    if mine is None or not theirs or theirs[0] is None:
        return False
    if mine[0] != theirs[0][0]:
        return False
    for key, value in (block.get("slots") or {}).items():
        if key in mine[1]:
            continue
        if key not in item.slots or str(item.slots[key]).casefold() != str(value).casefold():
            return False
    return True


def stock_blocks(item: LearnedIntent, body: dict[str, Any], data: dict[str, Any]) -> tuple[list[dict], list[str]]:
    """Data blocks of a learned sentence for a stock Home Assistant intent, and the sentences that fit none.

    Each stock block of the intent with the same shape gets a copy of the
    fitting sentences, inheriting its fixed slots, context constraints and
    response. A sentence without a target (no ``{name}``, ``{area}`` or ``{floor}``)
    only fits the blocks that take the area from the satellite context.
    """
    rules = data.get("expansion_rules") or {}
    blocks = []
    fitted: set[str] = set()
    for stock in body.get("data") or []:
        sentences = [s for s in item.sentences if _fits(item, s, stock, rules)]
        if not sentences:
            continue
        fitted.update(sentences)
        block = {key: copy.deepcopy(stock[key]) for key in _INHERITED_KEYS if key in stock}
        block["sentences"] = sentences
        block["slots"] = {**(stock.get("slots") or {}), **item.slots}
        block["metadata"] = {**(stock.get("metadata") or {}), LEARNED_ID_KEY: item.id}
        blocks.append(block)
    misfits = [s for s in item.sentences if s not in fitted]
    return blocks, misfits


def stock_learned(custom: dict[str, Any] | None, learned: Iterable[LearnedIntent]) -> list[LearnedIntent]:
    """Learned sentences for intents that neither the custom files nor a learned intent define.

    They go to the stock Home Assistant set; the others to the custom set.
    """
    learned = list(learned)
    names = set((custom or {}).get("intents") or {}) | {item.intent for item in learned if item.kind == "intent"}
    return [item for item in learned if item.kind == "sentence" and item.intent not in names]


def new_intents(learned: Iterable[LearnedIntent]) -> dict[str, LearnedIntent]:
    """Learned ``intent`` rows with a tool mapping, by intent name."""
    return {item.intent: item for item in learned if item.kind == "intent" and item.tool and item.sentences}


def tool_entries(learned: Iterable[LearnedIntent]) -> dict[str, dict[str, Any]]:
    """``client_tools.yaml`` entries of the learned intents."""
    return {name: copy.deepcopy(item.tool or {}) for name, item in new_intents(learned).items()}


def merge_learned(
    data: dict[str, Any], learned: Sequence[LearnedIntent], *, add_intents: bool, home_assistant: bool = False
) -> tuple[dict[str, Any], list[int]]:
    """Return a copy of hassil sentence ``data`` with the learned sentences merged in, and the ids used.

    A ``sentence`` row is merged when its intent is in ``data`` (or is a learned
    intent added here). ``add_intents`` adds the learned ``intent`` rows with their
    response template; only the custom sentence set takes them.
    ``home_assistant``: ``data`` is the stock set, sentences take the shape of its
    blocks (``stock_blocks``).
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
        if home_assistant:
            blocks, misfits = stock_blocks(item, body, merged)
            if misfits:
                logger.warning(
                    f"Intent engine: learned suggestion #{item.id} ignored: {'; '.join(misfits)} "
                    f"fits no {item.intent} sentence shape"
                )
                continue
        else:
            blocks = [item.data_block()]
        body["data"] = [*(body.get("data") or []), *blocks]
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

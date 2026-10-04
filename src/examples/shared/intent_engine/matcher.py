# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sentence matching for the intent engine, built on hassil.

Two sentence sets are tried in order: the custom files under
``<sentences_dir>/<language>/*.yaml`` and then the stock Home Assistant
sentences from the ``home-assistant-intents`` package, pruned to the allow-list.
A language with no data in either set never matches.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import home_assistant_intents
import yaml
from hassil import Intents, recognize_best
from hassil.intents import SlotList, TextSlotList
from hassil.util import merge_dict, remove_punctuation
from loguru import logger

# Leading words dropped before matching, in addition to the configured wake words.
# The stock sentences match polite forms but not a greeting or a name in front.
_LEADING_WORDS: dict[str, tuple[str, ...]] = {
    "en": ("hi", "hello", "hey", "ok", "okay", "thanks", "thank you"),
    "fr": ("bonjour", "salut", "coucou", "hey", "ok", "merci", "dis-moi", "dis moi", "dis"),
}


@dataclass(frozen=True)
class IntentMatch:
    """A recognized intent with the slot values to send to a target."""

    name: str
    slots: dict[str, Any]
    response_key: str = "default"
    custom: bool = False


@dataclass(frozen=True)
class _SentenceSet:
    intents: Intents
    responses: dict[str, dict[str, str]]
    errors: dict[str, str]


def base_language(code: str) -> str:
    """Return the base language of a BCP-47 code (``fr-FR`` -> ``fr``)."""
    return (code or "").split("-")[0].strip().lower()


def _language_candidates(code: str) -> list[str]:
    """Return lookup keys for a session language, most specific first."""
    code = (code or "").strip()
    return list(dict.fromkeys(key for key in (code, base_language(code)) if key))


def normalize_text(text: str, language: str, wake_words: Iterable[str] = ()) -> str:
    """Collapse whitespace and strip a leading greeting and/or wake word.

    ``"Bonjour Reachy, allume la lampe"`` becomes ``"allume la lampe"``. The
    prefix is only removed when something remains to match.
    """
    text = " ".join((text or "").split())
    greetings = _LEADING_WORDS.get(base_language(language), ())
    names = [word.strip() for word in wake_words if word.strip()]
    prefix = ""
    if greetings:
        prefix += rf"(?:(?:{'|'.join(re.escape(word) for word in greetings)})(?:[\s,.!]+|$))?"
    if names:
        prefix += rf"(?:(?:{'|'.join(re.escape(word) for word in names)})(?:[\s,.!:;-]+|$))?"
    if not prefix:
        return text
    stripped = re.sub(rf"^{prefix}", "", text, count=1, flags=re.IGNORECASE).strip()
    return stripped or text


def build_slot_lists(
    entities: Iterable[dict[str, Any]],
    areas: Iterable[str],
    floors: Iterable[str],
    blocked_domains: Iterable[str] = (),
) -> dict[str, SlotList]:
    """Build the hassil ``name``, ``area``, and ``floor`` lists from synced Home Assistant data.

    Entity names carry their ``domain`` (and ``device_class`` when set) as match
    context, which the stock sentences use in ``requires_context``. The slot value
    is the friendly name: several entities can share a name, and Home Assistant
    resolves the target by name when it handles the intent.

    Entities in ``blocked_domains`` are left out. So is any other entity that
    shares a blocked entity's name, because Home Assistant would act on both.
    """
    entities = list(entities)
    blocked = {str(domain).lower() for domain in blocked_domains}
    blocked_names = {
        str(entity.get("name") or "").strip().casefold()
        for entity in entities
        if str(entity.get("domain") or "").lower() in blocked
    }
    names: list[tuple[str, str, dict[str, Any]]] = []
    for entity in entities:
        name = str(entity.get("name") or "").strip()
        text = remove_punctuation(name).strip()
        if not text or name.casefold() in blocked_names:
            continue
        context: dict[str, Any] = {"domain": entity.get("domain")}
        if entity.get("device_class"):
            context["device_class"] = entity["device_class"]
        names.append((text, name, context))

    def _plain(values: Iterable[str]) -> list[tuple[str, str]]:
        pairs = ((remove_punctuation(str(value)).strip(), str(value)) for value in values if value)
        return [pair for pair in pairs if pair[0]]

    return {
        "name": TextSlotList.from_tuples(names, name="name", allow_template=False),
        "area": TextSlotList.from_tuples(_plain(areas), name="area", allow_template=False),
        "floor": TextSlotList.from_tuples(_plain(floors), name="floor", allow_template=False),
    }


_EMPTY_SLOT_LISTS = build_slot_lists([], [], [])


@lru_cache(maxsize=8)
def _stock_data(language: str) -> dict[str, Any] | None:
    """Return the packaged Home Assistant sentence data for a session language, if any."""
    for key in _language_candidates(language):
        data = home_assistant_intents.get_intents(key)
        if data:
            return data
    return None


@lru_cache(maxsize=8)
def _load_stock(language: str, allowed_intents: tuple[str, ...]) -> _SentenceSet | None:
    """Parse the stock sentences for a language, keeping only allow-listed intents.

    Pruning before parsing (instead of filtering matches) keeps an excluded
    wildcard intent from winning over the sentence the user actually said.
    """
    data = _stock_data(language)
    if not data:
        return None
    intents = {name: body for name, body in data.get("intents", {}).items() if name in allowed_intents}
    responses = data.get("responses", {})
    return _SentenceSet(
        intents=Intents.from_dict({**data, "intents": intents}),
        responses=responses.get("intents", {}),
        errors=responses.get("errors", {}),
    )


def _load_custom(sentences_dir: Path, language: str, keep: Callable[[str], bool]) -> _SentenceSet | None:
    """Parse the custom sentence files for a language, keeping intents accepted by ``keep``."""
    for key in _language_candidates(language):
        files = sorted((sentences_dir / key).glob("*.yaml")) if (sentences_dir / key).is_dir() else []
        if files:
            break
    else:
        return None

    merged: dict[str, Any] = {}
    for path in files:
        try:
            content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(f"Intent engine: ignoring unreadable sentence file {path}: {exc}")
            continue
        if isinstance(content, dict):
            merge_dict(merged, content)

    intents = {}
    for name, body in (merged.get("intents") or {}).items():
        if keep(name):
            intents[name] = body
        else:
            logger.debug(f"Intent engine: custom intent {name} is inactive for this session")
    if not intents:
        return None

    stock = _stock_data(language) or {}
    responses = merged.get("responses") or {}
    data = {
        "language": merged.get("language") or base_language(language),
        "intents": intents,
        "lists": merged.get("lists") or {},
        "expansion_rules": merged.get("expansion_rules") or {},
        # Reuse the stock polite-form skip words ("s'il te plaît", "peux-tu", ...).
        "skip_words": [*(stock.get("skip_words") or []), *(merged.get("skip_words") or [])],
    }
    try:
        parsed = Intents.from_dict(data)
    except Exception as exc:  # hassil raises several parse error types
        logger.warning(f"Intent engine: custom sentences for {language} failed to parse: {exc}")
        return None
    return _SentenceSet(intents=parsed, responses=responses.get("intents") or {}, errors=responses.get("errors") or {})


class IntentMatcher:
    """Match a user transcript against custom, then stock, sentences for one language."""

    def __init__(
        self,
        language: str,
        *,
        allowed_intents: Iterable[str],
        sentences_dir: Path,
        wake_words: Iterable[str] = (),
        area: str = "",
        keep_custom_intent: Callable[[str], bool] = lambda name: True,
        load_stock: bool = True,
    ):
        """Load the sentence sets for ``language`` (for example ``fr-FR``).

        ``load_stock=False`` skips the stock Home Assistant sentences, for a
        session that only has client tools.
        """
        self._language = language
        self._wake_words = tuple(wake_words)
        # Same shape Home Assistant passes for a satellite's area. Sentences that
        # declare ``requires_context: {area: {slot: true}}`` copy it into the slots.
        self._context = {"area": {"value": area, "text": area}} if area else None
        self._stock = _load_stock(language, tuple(allowed_intents)) if load_stock else None
        # Localized error sentences come from the language package even without its intents.
        self._packaged_errors = ((_stock_data(language) or {}).get("responses") or {}).get("errors") or {}
        self._custom = _load_custom(sentences_dir, language, keep_custom_intent)

    @property
    def has_data(self) -> bool:
        """Return whether any sentence set is available for the session language."""
        return self._stock is not None or self._custom is not None

    @property
    def custom_intent_names(self) -> list[str]:
        """Return the active custom intent names."""
        return sorted(self._custom.intents.intents) if self._custom else []

    def response_template(self, match_or_name: IntentMatch | str, response_key: str = "default") -> str | None:
        """Return the response template for an intent, custom files first."""
        if isinstance(match_or_name, IntentMatch):
            match_or_name, response_key = match_or_name.name, match_or_name.response_key
        for sentence_set in (self._custom, self._stock):
            if sentence_set and response_key in sentence_set.responses.get(match_or_name, {}):
                return sentence_set.responses[match_or_name][response_key]
        return None

    def error_template(self, key: str = "handle_error") -> str | None:
        """Return the localized error sentence for ``key``, custom files first."""
        if self._custom and key in self._custom.errors:
            return self._custom.errors[key]
        return self._packaged_errors.get(key)

    def match(self, text: str, slot_lists: dict[str, SlotList] | None = None) -> IntentMatch | None:
        """Return the best match for ``text``, or None.

        Stock Home Assistant sentences are only tried when ``slot_lists`` (the
        synced entity, area, and floor names) is provided.
        """
        text = normalize_text(text, self._language, self._wake_words)
        if not text:
            return None
        candidates = [(self._custom, slot_lists or _EMPTY_SLOT_LISTS, True)]
        if slot_lists is not None:
            candidates.append((self._stock, slot_lists, False))
        for sentence_set, lists, custom in candidates:
            if sentence_set is None:
                continue
            try:
                result = recognize_best(
                    text,
                    sentence_set.intents,
                    slot_lists=lists,
                    intent_context=self._context,
                    language=base_language(self._language),
                    best_slot_name="name",
                )
            except Exception as exc:  # a malformed custom sentence must not break the turn
                logger.warning(f"Intent engine: matching failed ({type(exc).__name__}: {exc})")
                continue
            if result is not None:
                return IntentMatch(
                    name=result.intent.name,
                    slots={entity.name: entity.value for entity in result.entities_list},
                    response_key=result.response or "default",
                    custom=custom,
                )
        return None

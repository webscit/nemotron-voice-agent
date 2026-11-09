# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Messaging sentences for the ``dream`` job: what may be learned, and what is never stored.

Turns where the LLM ran a messaging tool (``check_messages``, ``read_messages``,
``send_message``, with or without ``confirm_send_message``) teach sentences for
the request intents of the messaging dialog:

- ``check_messages`` -> ``MessagingCheck`` (no slot);
- ``read_messages`` -> ``MessagingRead`` (``{contact}`` when the call named one);
- ``send_message`` -> ``MessagingSend`` (``{contact}`` and a trailing ``{message}``)
  or ``MessagingReply`` (a trailing ``{message}``, the contact is the last sender).

A turn that only ran ``confirm_send_message`` answered a question ("yes"): the
dialog answers (``MessagingYes``, ``MessagingNo``, ``MessagingChange``) and the
capture steps (``MessagingDictate``, ``MessagingRecord``, ``MessagingContinue``)
are never learned.

Contact names and message text never reach a stored suggestion: names and aliases
in a proposal become ``{contact}`` and a proposal still naming a contact, or
copying words of the message, is dropped. Evidence keeps the sentence shape of
the person's words ("tell {contact} that {message}") and the tool names only.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from typing import Any

from loguru import logger

from examples.shared.intent_engine.learned import MESSAGING_SENTENCE_LISTS

CHECK_TOOL = "check_messages"
READ_TOOL = "read_messages"
SEND_TOOL = "send_message"
CONFIRM_TOOL = "confirm_send_message"
MESSAGING_TOOLS = frozenset({CHECK_TOOL, READ_TOOL, SEND_TOOL, CONFIRM_TOOL})
# The tool call a learned sentence of each intent must reproduce.
INTENT_TOOLS = {
    "MessagingCheck": CHECK_TOOL,
    "MessagingRead": READ_TOOL,
    "MessagingSend": SEND_TOOL,
    "MessagingReply": SEND_TOOL,
}
LEARNABLE_INTENTS = frozenset(MESSAGING_SENTENCE_LISTS)
CONTACT = "contact"
MESSAGE = "message"
CONTACT_PLACEHOLDER = "{contact}"
MESSAGE_PLACEHOLDER = "{message}"
# How close the captured {message} must be to the text the LLM sent (it may rephrase).
MESSAGE_SIMILARITY = 0.5
# Consecutive message words that a sentence may not contain.
_MESSAGE_WINDOW = 3
_SYNTAX_RE = re.compile(r"[\[\]()|<>]")
_LIST_RE = re.compile(r"\{[^}]*\}")
_WORD_RE = re.compile(r"\w+")

# What each messaging intent does, for the dream prompt.
DESCRIPTIONS = {
    "MessagingCheck": "the messaging dialog: says whether there are new messages (LLM tool check_messages)",
    "MessagingRead": "the messaging dialog: reads the new messages, of {contact} when named (LLM tool read_messages)",
    "MessagingSend": "the messaging dialog: prepares a message to {contact}, text {message} (LLM tool send_message)",
    "MessagingReply": "the messaging dialog: replies {message} to the last sender (LLM tool send_message)",
}


def _fold(text: str) -> str:
    from messaging.contacts import fold_name

    return fold_name(text)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_fold(text))


def load_contacts() -> list:
    """The Matrix contacts the server uses (``MATRIX_CONTACTS_FILE``); empty when unreadable."""
    try:
        from messaging.config import load_matrix_config
        from messaging.contacts import ContactBook

        return ContactBook(load_matrix_config().contacts_file).all()
    except Exception as exc:
        logger.warning(f"dream intents: messaging contacts unavailable ({type(exc).__name__}: {exc})")
        return []


def session_uses_messaging(turns: Iterable[dict[str, Any]]) -> bool:
    """Whether the session ran the messaging dialog: a messaging tool call or a messaging intent answered."""
    from examples.shared.intent_engine.learned import MESSAGING_INTENTS

    for turn in turns:
        if (turn.get("handled_by_intent") or "") in MESSAGING_INTENTS:
            return True
        if any(call.get("name") in MESSAGING_TOOLS for call in turn.get("tool_calls") or []):
            return True
    return False


class MessagingPolicy:
    """Contact names of a session and the messaging checks of the dream job."""

    def __init__(self, contacts: list, turns: Iterable[dict[str, Any]] = ()):
        """``contacts``: the contact book; ``turns``: the session's annotated turns (their tool-call contacts)."""
        self.contacts = list(contacts)
        names = {name for contact in self.contacts for name in contact.names()}
        self.spoken_names: set[str] = set()
        for turn in turns:
            for call in turn.get("tool_calls") or []:
                value = (call.get("arguments") or {}).get(CONTACT) if call.get("name") in MESSAGING_TOOLS else None
                if isinstance(value, str) and value.strip():
                    self.spoken_names.add(" ".join(value.split()))
        names |= self.spoken_names
        # Longest first: "Bob Smith" before "Bob".
        self.names = sorted({n for n in names if _fold(n).strip()}, key=lambda n: (-len(n), n))
        self.name_words = {w for n in self.names for w in _words(n) if len(w) >= 2}

    @property
    def available(self) -> bool:
        """Whether contact names are known: without them a name cannot be kept out of a sentence."""
        return bool(self.contacts)

    # ------------------------------------------------------------ slot list
    def slot_lists(self) -> dict[str, Any]:
        """The hassil ``contact`` list, as the server builds it, plus the names said in the tool calls."""
        from hassil.intents import TextSlotList
        from hassil.util import remove_punctuation

        pairs: list[tuple[str, str]] = []
        for contact in self.contacts:
            for name in contact.names():
                if text := remove_punctuation(name).strip():
                    pairs.append((text, contact.matrix_id))
        known = {_fold(text) for text, _ in pairs}
        for name in sorted(self.spoken_names):
            text = remove_punctuation(name).strip()
            if text and _fold(text) not in known:
                pairs.append((text, f"name:{_fold(text)}"))
        return {CONTACT: TextSlotList.from_tuples(pairs, name=CONTACT, allow_template=False)}

    def contact_ids(self, query: str) -> set[str]:
        """The ``contact`` slot values a tool call's contact argument stands for."""
        from messaging.contacts import match_contacts

        ids = {contact.matrix_id for contact in match_contacts(query, self.contacts)}
        return ids | {f"name:{_fold(query)}"}

    # ------------------------------------------------------------- sentences
    def _name_pattern(self, name: str) -> re.Pattern[str]:
        return re.compile(rf"(?<!\w){re.escape(_fold(name))}(?!\w)")

    def substitute(self, sentence: str) -> str:
        """``sentence`` with every contact name or alias written as ``{contact}`` (case and accents ignored)."""
        return _replace_folded(sentence, [self._name_pattern(n) for n in self.names], CONTACT_PLACEHOLDER)

    def name_problems(self, sentence: str) -> list[str]:
        """A sentence naming a contact outside of ``{contact}`` cannot be learned."""
        literal = _LIST_RE.sub(" ", sentence)
        if set(_words(_SYNTAX_RE.sub(" ", literal))) & self.name_words:
            return ["it names a contact; write {contact}"]
        return []

    @staticmethod
    def message_problems(sentence: str, messages: Iterable[str]) -> list[str]:
        """A sentence copying words of a message the person dictated cannot be learned."""
        words = _words(_SYNTAX_RE.sub(" ", _LIST_RE.sub(" ", sentence)))
        text = " " + " ".join(words) + " "
        for message in messages:
            body = _words(message)
            if not body or (len(body) == 1 and len(body[0]) < 4):
                continue  # "ok": too short to tell from the sentence's own words
            size = min(_MESSAGE_WINDOW, len(body))
            for start in range(len(body) - size + 1):
                if " " + " ".join(body[start : start + size]) + " " in text:
                    return ["it contains words of the message; write {message}"]
        return []

    # ---------------------------------------------------------------- checks
    def check_turn(self, intent: str, slots: dict[str, Any], tool_calls: list[dict[str, Any]]) -> str | None:
        """Why a match of ``intent`` with ``slots`` does not reproduce the turn's tool call (None when it does)."""
        tool = INTENT_TOOLS.get(intent)
        calls = [c for c in tool_calls if c.get("name") == tool]
        if tool is None or not calls:
            return f"no {tool or 'messaging'} call"
        for call in calls:
            arguments = call.get("arguments") or {}
            if self._same_call(intent, slots, arguments):
                return None
        return "the slots differ from the tool call"

    def _same_call(self, intent: str, slots: dict[str, Any], arguments: dict[str, Any]) -> bool:
        contact = str(arguments.get(CONTACT) or "").strip()
        if intent == "MessagingCheck":
            return not slots
        if intent == "MessagingRead":
            if not contact:
                return CONTACT not in slots
            return str(slots.get(CONTACT) or "") in self.contact_ids(contact)
        if intent == "MessagingSend" and (
            not contact or str(slots.get(CONTACT) or "") not in self.contact_ids(contact)
        ):
            return False
        return similar_message(str(slots.get(MESSAGE) or ""), str(arguments.get("text") or ""))

    # --------------------------------------------------------------- privacy
    def redact_text(self, text: str, slots: dict[str, Any], tool_calls: list[dict[str, Any]]) -> str | None:
        """The person's words with the message and contact names replaced by placeholders.

        None when the message cannot be located: the words are then not stored at all.
        """
        from examples.shared.intent_engine.matcher import original_span

        out = text
        captured = str(slots.get(MESSAGE) or "").strip()
        if captured:
            span = original_span(out, captured)
            if not isinstance(span, str) or span not in out:
                return None
            out = out.replace(span, MESSAGE_PLACEHOLDER)
        for call in tool_calls:
            body = str((call.get("arguments") or {}).get("text") or "").strip()
            if body:
                out = _replace_folded(out, [re.compile(re.escape(_fold(body)))], MESSAGE_PLACEHOLDER)
        out = self.substitute(out)
        if self.name_problems(out):
            return None
        return out

    @staticmethod
    def redact_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """The turn's messaging tool calls with placeholders for the contact and the message."""
        out = []
        for call in tool_calls:
            if call.get("name") not in MESSAGING_TOOLS:
                continue
            arguments = call.get("arguments") or {}
            redacted: dict[str, Any] = {}
            if arguments.get(CONTACT):
                redacted[CONTACT] = CONTACT_PLACEHOLDER
            if arguments.get("text"):
                redacted["text"] = MESSAGE_PLACEHOLDER
            if isinstance(arguments.get("include_read"), bool):
                redacted["include_read"] = arguments["include_read"]
            out.append({"name": call["name"], "arguments": redacted})
        return out or None


def similar_message(captured: str, sent: str) -> bool:
    """Whether the words a ``{message}`` captured are the message the LLM sent (it may rephrase a little)."""
    a, b = " ".join(_words(captured)), " ".join(_words(sent))
    if not a or not b:
        return False
    return a == b or a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= MESSAGE_SIMILARITY


def _replace_folded(text: str, patterns: list[re.Pattern[str]], placeholder: str) -> str:
    """Replace the matches of ``patterns`` (written for the folded text) in ``text``, first pattern first."""
    folded, origin = [], []
    for index, char in enumerate(text):
        for folded_char in _fold(char) or char:
            folded.append(folded_char)
            origin.append(index)
    folded_text = "".join(folded)
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        for found in pattern.finditer(folded_text):
            if found.end() <= found.start():
                continue
            start, end = origin[found.start()], origin[found.end() - 1] + 1
            if all(end <= s or start >= e for s, e in spans):
                spans.append((start, end))
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + placeholder + text[end:]
    return text

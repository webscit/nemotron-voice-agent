# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix messaging as intent-engine intents: a deterministic dialog, without the LLM.

The sentences and the spoken replies live in the intent-engine sentence files
(``sentences/<language>/messaging.yaml``); replies are rendered from the
``Messaging`` response templates. Every step asks at most one question, answered
in the next turn through the engine's follow-up:

- "Do I have messages?" -> "You have 2 new messages from Bob. Shall I read them?" -> yes -> read.
- "Read my messages [from Bob]" -> read, then "Do you want to reply to Bob?" -> yes -> dictation.
- "Send a message to Bob" -> "What should I tell Bob?" -> <the user's words> -> read back,
  "Shall I send it?" -> yes -> sent / no -> dropped / "change it" -> dictation again.
- "Tell Bob that <message>" goes straight to the read-back.

The checks of ``MessagingTools`` (contact owners, the person known for sure,
the draft) apply unchanged: the engine and the LLM tools share one instance, so
a draft made by either one is the same draft.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from hassil.intents import SlotList, TextSlotList
from hassil.util import remove_punctuation
from jinja2 import TemplateError
from loguru import logger

from examples.shared.intent_engine.responses import render_local
from examples.shared.intent_engine.targets import (
    FollowUp,
    IntentRequest,
    IntentResponse,
    SpeechTemplates,
    TargetUnavailable,
)
from examples.shared.messaging_tools import MessagingTools
from messaging.contacts import Contact
from messaging.inbox import KIND_AUDIO, InboxMessage

CHECK_INTENT = "MessagingCheck"
READ_INTENT = "MessagingRead"
SEND_INTENT = "MessagingSend"
REPLY_INTENT = "MessagingReply"
YES_INTENT = "MessagingYes"
NO_INTENT = "MessagingNo"
CHANGE_INTENT = "MessagingChange"
# Not in the sentence files: the engine routes dictated words here.
DICTATE_INTENT = "MessagingDictate"
MESSAGING_INTENTS = frozenset(
    {CHECK_INTENT, READ_INTENT, SEND_INTENT, REPLY_INTENT, YES_INTENT, NO_INTENT, CHANGE_INTENT, DICTATE_INTENT}
)

# Hassil intent context of a pending messaging question (``requires_context: {dialog: messaging}``).
DIALOG_CONTEXT = "messaging"
# Response templates: ``responses.intents.Messaging.<key>``.
RESPONSES = "Messaging"
CONTACT_SLOT = "contact"
MESSAGE_SLOT = "message"

# Pending questions.
OFFER_READ = "offer_read"
OFFER_MORE = "offer_more"
OFFER_REPLY = "offer_reply"
DICTATION = "dictation"
CONFIRM = "confirm"

# Error ``code`` of ``MessagingTools`` -> response key.
_ERROR_KEYS = {
    "unknown_person": "unknown_person",
    "not_allowed": "not_allowed",
    "no_contacts": "no_contact",
    "no_contact": "no_contact",
    "ambiguous": "no_contact",
    "too_long": "too_long",
    "no_draft": "no_draft",
    "send_failed": "send_failed",
}


def message_slots(message: InboxMessage, contact: Contact, now: datetime | None = None) -> dict[str, Any]:
    """Template slots of a received message; the template says when it arrived in its own language."""
    sent = datetime.fromtimestamp(message.sent_at)
    now = now or datetime.now()
    return {
        "sender": contact.name,
        "text": "" if message.kind == KIND_AUDIO else message.body,
        "voice": message.kind == KIND_AUDIO,
        "time": f"{sent:%H:%M}",
        "today": sent.date() == now.date(),
        "yesterday": sent.date() == (now - timedelta(days=1)).date(),
        "weekday": sent.weekday(),
        "day": sent.day,
        "month": sent.month,
    }


def contact_slot_list(contacts: list[Contact]) -> TextSlotList:
    """The hassil ``contact`` list: every name and alias, valued by Matrix id.

    The matcher respells names said with other accents or case (``respell_slot_values``).
    """
    pairs: list[tuple[str, str]] = []
    for contact in contacts:
        for name in contact.names():
            if text := remove_punctuation(name).strip():
                pairs.append((text, contact.matrix_id))
    return TextSlotList.from_tuples(pairs, name=CONTACT_SLOT, allow_template=False)


class MessagingIntents:
    """Server target running the messaging dialog for one session."""

    name = "messaging"
    # A send waits out matrix.org rate limits; the engine falls back to the LLM after this.
    timeout_secs = 15.0

    def __init__(self, tools: MessagingTools, contacts: Callable[[], list[Contact]]):
        """Run the dialog on ``tools``; ``contacts`` returns every contact (for the slot list)."""
        self._tools = tools
        self._contacts = contacts
        self._templates: SpeechTemplates | None = None
        self._lists_key: tuple[Contact, ...] | None = None
        self._lists: dict[str, SlotList] = {}
        # The pending question and what it is about.
        self._state: str | None = None
        self._contact: Contact | None = None
        self._read_contacts: list[Contact] = []
        # Who wrote the messages just read (one sender): "reply to them".
        self._reply_to: Contact | None = None

    # ------------------------------------------------------- ServerTarget
    @property
    def intent_names(self) -> frozenset[str]:
        """Return the messaging intents."""
        return MESSAGING_INTENTS

    def bind_templates(self, templates: SpeechTemplates) -> None:
        """Use ``templates`` (the session's matcher) for the replies."""
        self._templates = templates

    def slot_lists(self) -> dict[str, SlotList]:
        """Return the ``contact`` list, rebuilt when the contacts file changed."""
        contacts = tuple(self._contacts())
        if contacts != self._lists_key:
            self._lists_key = contacts
            self._lists = {CONTACT_SLOT: contact_slot_list(list(contacts))}
        return self._lists

    def build_call(self, request: IntentRequest) -> dict[str, Any]:
        """Describe the step a request would run (dry-run log; message text left out)."""
        return {"intent": request.name, "contact": request.data.get(CONTACT_SLOT), "state": self._state}

    async def handle(self, request: IntentRequest) -> IntentResponse:
        """Run one dialog step."""
        slots = request.data
        state, self._state = self._state, None
        if request.name == CHECK_INTENT:
            return await self._check()
        if request.name == READ_INTENT:
            if slots.get(CONTACT_SLOT):
                contact = self._tools.resolve(str(slots[CONTACT_SLOT]))
                if isinstance(contact, dict):
                    return self._error(contact)
                return await self._read([contact], one_contact=True)
            return await self._read(self._tools.allowed_contacts())
        if request.name == SEND_INTENT:
            contact = self._tools.resolve(str(slots.get(CONTACT_SLOT) or ""))
            if isinstance(contact, dict):
                return self._error(contact)
            return self._compose(contact, slots.get(MESSAGE_SLOT))
        if request.name == REPLY_INTENT:
            if self._reply_to is None:
                raise TargetUnavailable("nobody to reply to")
            return self._compose(self._reply_to, slots.get(MESSAGE_SLOT))
        if request.name == DICTATE_INTENT:
            if state != DICTATION or self._contact is None:
                raise TargetUnavailable("no dictation pending")
            return self._draft(self._contact, str(slots.get(MESSAGE_SLOT) or ""))
        if request.name == YES_INTENT:
            return await self._yes(state)
        if request.name == NO_INTENT:
            if state in (CONFIRM, DICTATION):
                self._tools.cancel_draft()
                return self._reply("cancelled")
            return self._reply("ok")
        if request.name == CHANGE_INTENT:
            draft = self._tools.draft
            if state != CONFIRM or draft is None:
                raise TargetUnavailable("no draft to change")
            return self._compose(draft.contact, slots.get(MESSAGE_SLOT))
        raise TargetUnavailable(f"unknown messaging intent {request.name}")

    # ------------------------------------------------------------- steps
    async def _check(self) -> IntentResponse:
        try:
            unread = await self._tools.unread()
        except Exception as exc:
            logger.opt(exception=exc).warning("Matrix: could not count unread messages")
            return self._reply("read_failed")
        if not unread:
            return self._reply("no_new")
        senders = [contact for contact, _ in unread]
        self._read_contacts = senders
        return self._reply(
            "unread",
            {"count": sum(n for _, n in unread), "senders": [c.name for c in senders]},
            ask=OFFER_READ,
        )

    async def _read(self, contacts: list[Contact], *, one_contact: bool = False) -> IntentResponse:
        try:
            messages, more = await self._tools.fetch(contacts)
        except Exception as exc:
            logger.opt(exception=exc).warning("Matrix: could not read messages")
            return self._reply("read_failed")
        if not messages:
            if one_contact:
                return self._reply("no_new_from", {CONTACT_SLOT: contacts[0].name})
            return self._reply("no_new")
        senders = list(dict.fromkeys(contact for _, contact in messages))
        self._reply_to = senders[0] if len(senders) == 1 else None
        self._read_contacts = contacts
        slots = {
            "messages": [message_slots(message, contact) for message, contact in messages],
            "more": more,
            "reply_to": self._reply_to.name if self._reply_to and not more else None,
        }
        if more:
            return self._reply("messages", slots, ask=OFFER_MORE)
        if self._reply_to is not None:
            return self._reply("messages", slots, ask=OFFER_REPLY, contact=self._reply_to)
        return self._reply("messages", slots)

    def _compose(self, contact: Contact, message: Any) -> IntentResponse:
        """Draft ``message`` for ``contact``, or ask for the words first."""
        if message and str(message).strip():
            return self._draft(contact, str(message))
        return self._reply("ask_text", {CONTACT_SLOT: contact.name}, ask=DICTATION, contact=contact)

    def _draft(self, contact: Contact, text: str) -> IntentResponse:
        result = self._tools.prepare(contact, text)
        if result.get("code") == "no_text":
            return self._reply("ask_text", {CONTACT_SLOT: contact.name}, ask=DICTATION, contact=contact)
        if "error" in result:
            return self._error(result)
        return self._reply("confirm", {CONTACT_SLOT: contact.name, MESSAGE_SLOT: result["text"]}, ask=CONFIRM)

    async def _yes(self, state: str | None) -> IntentResponse:
        if state in (OFFER_READ, OFFER_MORE):
            return await self._read(self._read_contacts or self._tools.allowed_contacts())
        if state == OFFER_REPLY and self._contact is not None:
            return self._compose(self._contact, None)
        if state == CONFIRM:
            draft = self._tools.draft
            result = await self._tools.send_draft(require_new_turn=False)
            if "error" in result:
                return self._error(result)
            return self._reply("sent", {CONTACT_SLOT: draft.contact.name if draft else result.get("to")})
        raise TargetUnavailable("no messaging question pending")

    # ----------------------------------------------------------- replies
    def _error(self, result: dict[str, Any]) -> IntentResponse:
        return self._reply(_ERROR_KEYS.get(str(result.get("code")), "failed"), response_type="error")

    def _render(self, key: str, slots: dict[str, Any]) -> str:
        template = self._templates.response_template(RESPONSES, key) if self._templates else None
        if not template:
            raise TargetUnavailable(f"no {RESPONSES}.{key} response for this language")
        try:
            return render_local(template, slots)
        except TemplateError as exc:
            raise TargetUnavailable(f"response {RESPONSES}.{key} failed: {exc}") from exc

    def _reply(
        self,
        key: str,
        slots: dict[str, Any] | None = None,
        *,
        ask: str | None = None,
        contact: Contact | None = None,
        response_type: str = "action_done",
    ) -> IntentResponse:
        """Render reply ``key``; ``ask`` makes it a question answered by the next turn."""
        slots = slots or {}
        speech = self._render(key, slots)
        self._state = ask
        self._contact = contact
        follow_up = None
        if ask == DICTATION:
            # Any words are the message, except "cancel".
            follow_up = FollowUp(
                context=DIALOG_CONTEXT,
                capture=MESSAGE_SLOT,
                capture_intent=DICTATE_INTENT,
                accept=frozenset({NO_INTENT}),
            )
        elif ask is not None:
            follow_up = FollowUp(
                context=DIALOG_CONTEXT,
                accept=frozenset({YES_INTENT, NO_INTENT, CHANGE_INTENT}),
                reprompt=self._render(f"reprompt_{ask}", slots),
            )
        return IntentResponse(speech=speech, response_type=response_type, speech_slots=slots, follow_up=follow_up)

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
- Voice messages: received ones play inside "read my messages", or with "play the
  voice message from Bob". "Send a voice message to Bob" -> "I'm listening" -> the
  next turn's audio is the message -> "Shall I send it?" -> yes / no / "play it" /
  "start over" (record again).

A yes/no answer may open a longer turn ("Oui. Lis-les."): the leading answer
counts. Words said over the read-back of a draft, before "Shall I send it?" was
heard, are the rest of the message: it is read back again, whole.

When the LLM runs a send step through its tools instead (``send_message``,
``confirm_send_message``), the outcome is still spoken from these templates
(``announce``), and the read-back question is answered by the engine: what the
agent says about a message being sent is always the gateway's result.

The checks of ``MessagingTools`` (contact owners, the person known for sure,
the draft) apply unchanged: the engine and the LLM tools share one instance, so
a draft made by either one is the same draft.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from hassil.intents import SlotList, TextSlotList
from hassil.util import remove_punctuation
from jinja2 import TemplateError
from loguru import logger

from examples.shared.intent_engine.responses import render_local
from examples.shared.intent_engine.targets import (
    AudioClip,
    FollowUp,
    IntentRequest,
    IntentResponse,
    SpeechTemplates,
    TargetUnavailable,
)
from examples.shared.messaging_tools import CONFIRM_TOOL_NAME, SEND_TOOL_NAME, MessagingTools, VoiceRecording
from messaging.contacts import Contact
from messaging.gateway import MatrixError
from messaging.inbox import KIND_AUDIO, InboxMessage

CHECK_INTENT = "MessagingCheck"
READ_INTENT = "MessagingRead"
SEND_INTENT = "MessagingSend"
REPLY_INTENT = "MessagingReply"
YES_INTENT = "MessagingYes"
NO_INTENT = "MessagingNo"
CHANGE_INTENT = "MessagingChange"
SEND_VOICE_INTENT = "MessagingSendVoice"
PLAY_INTENT = "MessagingPlay"
# Not in the sentence files: the engine routes dictated words (and the recorded turn) here,
# and words said over a read-back.
DICTATE_INTENT = "MessagingDictate"
RECORD_INTENT = "MessagingRecord"
CONTINUE_INTENT = "MessagingContinue"
MESSAGING_INTENTS = frozenset(
    {
        CHECK_INTENT,
        READ_INTENT,
        SEND_INTENT,
        REPLY_INTENT,
        YES_INTENT,
        NO_INTENT,
        CHANGE_INTENT,
        SEND_VOICE_INTENT,
        PLAY_INTENT,
        DICTATE_INTENT,
        RECORD_INTENT,
        CONTINUE_INTENT,
    }
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
RECORDING = "recording"
CONFIRM = "confirm"

# Where a template plays clip <n> (``slots.messages[i].play``); the reply is split there.
_CLIP_MARK = "⟦clip{}⟧"
_CLIP_MARK_RE = re.compile(r"⟦clip(\d+)⟧")

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
    "voice_too_long": "voice_too_long",
}


def message_slots(
    message: InboxMessage, contact: Contact, now: datetime | None = None, *, play: str = ""
) -> dict[str, Any]:
    """Template slots of a received message; the template says when it arrived in its own language.

    ``play`` is the mark where a voice message plays (empty when it cannot be played).
    """
    sent = datetime.fromtimestamp(message.sent_at)
    now = now or datetime.now()
    return {
        "sender": contact.name,
        "text": "" if message.kind == KIND_AUDIO else message.body,
        "voice": message.kind == KIND_AUDIO,
        "play": play,
        "seconds": max(1, round(message.duration_ms / 1000)) if message.duration_ms else 0,
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

    def __init__(
        self,
        tools: MessagingTools,
        contacts: Callable[[], list[Contact]],
        *,
        voice_capture: Callable[[], AudioClip | None] | None = None,
        playback_sample_rate: int = 22050,
    ):
        """Run the dialog on ``tools``; ``contacts`` returns every contact (for the slot list).

        ``voice_capture`` returns the audio of the turn being handled (``UserTurnAudioCapture``);
        without it, voice messages cannot be recorded. Received ones are decoded at
        ``playback_sample_rate``.
        """
        self._tools = tools
        self._contacts = contacts
        self._voice_capture = voice_capture
        self._playback_sample_rate = playback_sample_rate
        self._templates: SpeechTemplates | None = None
        self._open_dialog: Callable[[FollowUp], None] | None = None
        self._lists_key: tuple[Contact, ...] | None = None
        self._lists: dict[str, SlotList] = {}
        # The pending question and what it is about.
        self._state: str | None = None
        self._contact: Contact | None = None
        self._read_contacts: list[Contact] = []
        # Who wrote the messages just read (one sender): "reply to them".
        self._reply_to: Contact | None = None
        tools.announcer = self.announce

    # ------------------------------------------------------- ServerTarget
    @property
    def intent_names(self) -> frozenset[str]:
        """Return the messaging intents."""
        return MESSAGING_INTENTS

    def bind_templates(self, templates: SpeechTemplates) -> None:
        """Use ``templates`` (the session's matcher) for the replies."""
        self._templates = templates

    def bind_dialog(self, open_dialog: Callable[[FollowUp], None]) -> None:
        """Use ``open_dialog`` to ask the read-back question after an LLM send step."""
        self._open_dialog = open_dialog

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
        if request.name == CONTINUE_INTENT:
            draft = self._tools.draft
            if state != CONFIRM or draft is None or draft.voice is not None:
                raise TargetUnavailable("no text draft to continue")
            return self._draft(draft.contact, f"{draft.text} {slots.get(MESSAGE_SLOT) or ''}")
        if request.name == SEND_VOICE_INTENT:
            if slots.get(CONTACT_SLOT):
                contact = self._tools.resolve(str(slots[CONTACT_SLOT]))
                if isinstance(contact, dict):
                    return self._error(contact)
            elif self._reply_to is not None:
                contact = self._reply_to
            else:
                raise TargetUnavailable("nobody to send a voice message to")
            return self._ask_voice(contact)
        if request.name == RECORD_INTENT:
            if state != RECORDING or self._contact is None:
                raise TargetUnavailable("no recording pending")
            return self._record(self._contact, str(slots.get(MESSAGE_SLOT) or ""))
        if request.name == PLAY_INTENT:
            draft = self._tools.draft
            if state == CONFIRM and draft is not None and draft.voice is not None and not slots.get(CONTACT_SLOT):
                return self._replay(draft.contact, draft.voice)
            if slots.get(CONTACT_SLOT):
                contact = self._tools.resolve(str(slots[CONTACT_SLOT]))
                if isinstance(contact, dict):
                    return self._error(contact)
                return await self._play_latest([contact])
            return await self._play_latest(self._tools.allowed_contacts())
        if request.name == YES_INTENT:
            return await self._yes(state)
        if request.name == NO_INTENT:
            if state in (CONFIRM, DICTATION, RECORDING):
                self._tools.cancel_draft()
                return self._reply("cancelled")
            return self._reply("ok")
        if request.name == CHANGE_INTENT:
            draft = self._tools.draft
            if state != CONFIRM or draft is None:
                raise TargetUnavailable("no draft to change")
            if draft.voice is not None and not slots.get(MESSAGE_SLOT):
                return self._ask_voice(draft.contact)
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
        clips = await self._voice_clips([message for message, _ in messages])
        marks = {index: _CLIP_MARK.format(n) for n, index in enumerate(sorted(clips))}
        slots = {
            "messages": [
                message_slots(message, contact, play=marks.get(index, ""))
                for index, (message, contact) in enumerate(messages)
            ],
            "more": more,
            "reply_to": self._reply_to.name if self._reply_to and not more else None,
        }
        played = [clips[index] for index in sorted(clips)]
        if more:
            return self._reply("messages", slots, ask=OFFER_MORE, clips=played)
        if self._reply_to is not None:
            return self._reply("messages", slots, ask=OFFER_REPLY, contact=self._reply_to, clips=played)
        return self._reply("messages", slots, clips=played)

    async def _voice_clips(self, messages: list[InboxMessage]) -> dict[int, AudioClip]:
        """The playable audio of the voice messages among ``messages``, by index (failures left out)."""
        indexes = [i for i, message in enumerate(messages) if message.kind == KIND_AUDIO]
        rate = self._playback_sample_rate
        results = await asyncio.gather(
            *(self._tools.voice_audio(messages[i], rate) for i in indexes), return_exceptions=True
        )
        clips: dict[int, AudioClip] = {}
        for index, result in zip(indexes, results, strict=True):
            if isinstance(result, MatrixError):
                logger.warning(f"Matrix: cannot play voice message {messages[index].event_id}: {result}")
            elif isinstance(result, BaseException):
                raise result
            elif result:
                clips[index] = AudioClip(result, rate)
        return clips

    async def _play_latest(self, contacts: list[Contact]) -> IntentResponse:
        try:
            found = await self._tools.latest_voice(contacts)
        except Exception as exc:
            logger.opt(exception=exc).warning("Matrix: could not read messages")
            return self._reply("read_failed")
        if found is None:
            name = contacts[0].name if len(contacts) == 1 else None
            return self._reply("no_voice_message", {CONTACT_SLOT: name})
        message, contact = found
        clips = await self._voice_clips([message])
        self._reply_to = contact
        slots = {
            "messages": [message_slots(message, contact, play=_CLIP_MARK.format(0) if clips else "")],
            "more": False,
            "reply_to": contact.name,
        }
        return self._reply("messages", slots, ask=OFFER_REPLY, contact=contact, clips=list(clips.values()))

    def _compose(self, contact: Contact, message: Any) -> IntentResponse:
        """Draft ``message`` for ``contact``, or ask for the words first."""
        if message and str(message).strip():
            return self._draft(contact, str(message))
        return self._reply("ask_text", {CONTACT_SLOT: contact.name}, ask=DICTATION, contact=contact)

    def _ask_voice(self, contact: Contact) -> IntentResponse:
        if self._voice_capture is None:
            raise TargetUnavailable("voice messages need the microphone audio")
        return self._reply("ask_voice", {CONTACT_SLOT: contact.name}, ask=RECORDING, contact=contact)

    def _record(self, contact: Contact, transcript: str) -> IntentResponse:
        """Draft the turn just heard as a voice message for ``contact``."""
        clip = self._voice_capture() if self._voice_capture else None
        if clip is None:
            return self._reply("no_voice", {CONTACT_SLOT: contact.name}, ask=RECORDING, contact=contact)
        result = self._tools.prepare_voice(contact, VoiceRecording(clip.audio, clip.sample_rate), transcript)
        if result.get("code") == "no_voice":
            return self._reply("no_voice", {CONTACT_SLOT: contact.name}, ask=RECORDING, contact=contact)
        if "error" in result:
            return self._error(result)
        return self._reply("confirm_voice", {CONTACT_SLOT: contact.name, "seconds": result["seconds"]}, ask=CONFIRM)

    def _replay(self, contact: Contact, voice: VoiceRecording) -> IntentResponse:
        slots = {CONTACT_SLOT: contact.name, "play": _CLIP_MARK.format(0)}
        return self._reply("replay_voice", slots, ask=CONFIRM, clips=[AudioClip(voice.audio, voice.sample_rate)])

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

    # ------------------------------------------------------ LLM send steps
    def announce(self, tool: str, result: dict[str, Any]) -> str | None:
        """What to say about the result of the LLM tool ``tool``, or None to let the LLM say it.

        A draft is read back and its question left to the engine; a send says what
        the gateway answered. Other tools and drafting errors stay with the LLM.
        """
        if self._templates is None or self._open_dialog is None:
            return None
        draft = self._tools.draft
        try:
            if tool == SEND_TOOL_NAME and "error" not in result:
                response = self._reply(
                    "confirm", {CONTACT_SLOT: result["to"], MESSAGE_SLOT: result["text"]}, ask=CONFIRM
                )
            elif tool == CONFIRM_TOOL_NAME and result.get("code") == "not_confirmed" and draft is not None:
                response = self._reply("reprompt_confirm", {CONTACT_SLOT: draft.contact.name}, ask=CONFIRM)
            elif tool == CONFIRM_TOOL_NAME and "error" in result:
                response = self._error(result)
            elif tool == CONFIRM_TOOL_NAME:
                response = self._reply("sent", {CONTACT_SLOT: result.get("to")})
            else:
                return None
        except TargetUnavailable as exc:
            logger.warning(f"Matrix: no spoken reply for {tool} ({exc}); the LLM answers")
            return None
        if response.follow_up is not None:
            self._open_dialog(response.follow_up)
        return response.speech

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
        clips: list[AudioClip] | None = None,
    ) -> IntentResponse:
        """Render reply ``key``; ``ask`` makes it a question answered by the next turn.

        ``clips`` play where the rendered reply has their marks (``_CLIP_MARK``).
        """
        slots = slots or {}
        speech = self._render(key, slots)
        self._state = ask
        self._contact = contact
        follow_up = None
        if ask in (DICTATION, RECORDING):
            # Any words are the message (or the recording), except "cancel".
            follow_up = FollowUp(
                context=DIALOG_CONTEXT,
                capture=MESSAGE_SLOT,
                capture_intent=DICTATE_INTENT if ask == DICTATION else RECORD_INTENT,
                accept=frozenset({NO_INTENT}),
            )
        elif ask is not None:
            draft = self._tools.draft
            continues = ask == CONFIRM and draft is not None and draft.voice is None
            follow_up = FollowUp(
                context=DIALOG_CONTEXT,
                accept=frozenset({YES_INTENT, NO_INTENT, CHANGE_INTENT, PLAY_INTENT}),
                reprompt=self._render(f"reprompt_{ask}", slots),
                leading=frozenset({YES_INTENT, NO_INTENT}),
                # Said over the read-back of a text draft: the message goes on.
                continue_intent=CONTINUE_INTENT if continues else None,
                continue_slot=MESSAGE_SLOT if continues else None,
            )
        parts = _split_clips(speech, clips or [])
        if _CLIP_MARK_RE.search(speech):
            speech = " ".join(_CLIP_MARK_RE.sub(" ", speech).split())
        return IntentResponse(
            speech=speech,
            response_type=response_type,
            speech_slots=slots,
            follow_up=follow_up,
            parts=parts,
        )


def _split_clips(speech: str, clips: list[AudioClip]) -> tuple[str | AudioClip, ...]:
    """``speech`` cut at its clip marks, with the clips in between; () when it plays none."""
    if not _CLIP_MARK_RE.search(speech):
        return ()
    parts: list[str | AudioClip] = []
    for n, piece in enumerate(_CLIP_MARK_RE.split(speech)):
        if n % 2 == 0:
            if piece.strip():
                parts.append(" ".join(piece.split()))
        elif int(piece) < len(clips):
            parts.append(clips[int(piece)])
    return tuple(parts) if any(isinstance(part, AudioClip) for part in parts) else ()

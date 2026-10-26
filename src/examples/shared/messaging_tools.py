# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side Matrix messaging tools: check, read and send messages to the agent's contacts.

Messages go out from the agent's own Matrix account, signed with the name of the
person who asked for them. A contact with an ``owner`` is only usable by that
person while they are known for sure (picked in the client, or recognised with a
trusted voice ID tier); contacts without an owner are shared.

Sending takes two steps: ``send_message`` only prepares a draft that the agent
reads back, and ``confirm_send_message`` sends it once the user has answered,
so a misheard name or text never leaves the house unconfirmed.

When the intent engine runs, it speaks the outcome of these two tools itself
(``announcer``) and the LLM is not run on their result: the read-back is the
draft as stored, and "sent" is only said once the gateway sent it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)

from examples.shared.person_memory import PersonContext
from messaging.contacts import Contact, match_contacts
from messaging.gateway import MatrixError, MatrixGateway
from messaging.inbox import KIND_AUDIO, InboxMessage

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

CHECK_TOOL_NAME = "check_messages"
READ_TOOL_NAME = "read_messages"
SEND_TOOL_NAME = "send_message"
CONFIRM_TOOL_NAME = "confirm_send_message"
MESSAGING_TOOL_NAMES = (CHECK_TOOL_NAME, READ_TOOL_NAME, SEND_TOOL_NAME, CONFIRM_TOOL_NAME)
MESSAGING_ADDON_KEY = "messaging_addon"
MAX_MESSAGE_LEN = 2000
MAX_READ_MESSAGES = 5
DRAFT_TTL_SECS = 300.0


def _schemas() -> list[FunctionSchema]:
    contact_hint = "The contact's name, as the user said it."
    return [
        FunctionSchema(
            name=CHECK_TOOL_NAME,
            description=(
                "Check whether there are new (unread) messages from the user's contacts, without reading them. "
                "Use it when the user asks if they have messages."
            ),
            properties={},
            required=[],
        ),
        FunctionSchema(
            name=READ_TOOL_NAME,
            description=(
                "Get the user's new messages so you can read them aloud, oldest first. They count as read once "
                "returned. Set include_read to repeat the latest messages of a contact the user already heard."
            ),
            properties={
                "contact": {"type": "string", "description": f"{contact_hint} Leave empty for every contact."},
                "include_read": {
                    "type": "boolean",
                    "description": "Return the latest messages even if they were already read.",
                },
            },
            required=[],
        ),
        FunctionSchema(
            name=SEND_TOOL_NAME,
            description=(
                "Prepare a text message to one of the user's contacts. It is NOT sent yet: the user must confirm "
                "it first. When the result has 'spoken', the user already heard the read-back and the question; "
                "otherwise read the recipient and the text back and ask them to confirm. Call it again to change "
                "the draft."
            ),
            properties={
                "contact": {"type": "string", "description": contact_hint},
                "text": {
                    "type": "string",
                    "description": "The message, in the user's words and language, written as they want it sent.",
                },
            },
            required=["contact", "text"],
        ),
        FunctionSchema(
            name=CONFIRM_TOOL_NAME,
            description=(
                "Send the message prepared with send_message. Call it only after the user clearly confirmed the "
                "draft you read back to them; if they declined, do not call it."
            ),
            properties={},
            required=[],
        ),
    ]


def with_messaging_tools(tools: object) -> ToolsSchema:
    """``tools`` (a ``ToolsSchema`` or not given) plus the messaging tools."""
    schemas = _schemas()
    if isinstance(tools, ToolsSchema):
        standard = [t for t in tools.standard_tools if getattr(t, "name", None) not in MESSAGING_TOOL_NAMES]
        return ToolsSchema(standard_tools=[*standard, *schemas], custom_tools=tools.custom_tools)
    return ToolsSchema(standard_tools=schemas)


def spoken_time(timestamp: float, now: datetime | None = None) -> str:
    """When a message arrived, the way it is said aloud ("today 14:05", "yesterday 09:30", "Monday 12 October")."""
    sent = datetime.fromtimestamp(timestamp)
    now = now or datetime.now()
    if sent.date() == now.date():
        return f"today {sent:%H:%M}"
    if sent.date() == (now - timedelta(days=1)).date():
        return f"yesterday {sent:%H:%M}"
    return f"{sent:%A} {sent.day} {sent:%B}"


def describe_message(message: InboxMessage, contact_name: str) -> dict[str, Any]:
    """The tool-result form of a received message."""
    result: dict[str, Any] = {"from": contact_name, "when": spoken_time(message.sent_at)}
    if message.kind == KIND_AUDIO:
        result["voice_message"] = True
        result["note"] = "This is a voice message; you cannot play it yet. Tell the user it is waiting in Element."
    else:
        result["text"] = message.body
    return result


@dataclass(frozen=True)
class Draft:
    """A message waiting for confirmation."""

    contact: Contact
    text: str
    sender_name: str | None
    person_id: str | None
    user_turn: int
    created_at: float


class MessagingTools:
    """Messaging tool handlers for one live session."""

    def __init__(
        self,
        *,
        gateway: MatrixGateway,
        current_person: Callable[[], PersonContext | None],
        user_turn: Callable[[], int],
    ):
        """Bind the tools to the session.

        Args:
            gateway: The process-wide Matrix gateway.
            current_person: The person known for sure, or None.
            user_turn: Number of user turns so far (a draft is confirmed in a later turn than it was made).
        """
        self._gateway = gateway
        self._current_person = current_person
        self._user_turn = user_turn
        self._draft: Draft | None = None
        # Speech for the result of a send step, or None to let the LLM speak (set by ``MessagingIntents``).
        self.announcer: Callable[[str, dict[str, Any]], str | None] | None = None

    # ------------------------------------------------------------ handlers
    async def check_messages(self, params: FunctionCallParams) -> None:
        """Server-side ``check_messages()`` tool handler."""
        await params.result_callback(await self._check())

    async def read_messages(self, params: FunctionCallParams) -> None:
        """Server-side ``read_messages(contact?, include_read?)`` tool handler."""
        args = params.arguments
        query = " ".join(str(args.get("contact") or "").split())
        await params.result_callback(await self._read(query, bool(args.get("include_read"))))

    async def send_message(self, params: FunctionCallParams) -> None:
        """Server-side ``send_message(contact, text)`` tool handler: prepares a draft."""
        args = params.arguments
        query = " ".join(str(args.get("contact") or "").split())
        await self._finish(params, SEND_TOOL_NAME, self._prepare(query, str(args.get("text") or "").strip()))

    async def confirm_send_message(self, params: FunctionCallParams) -> None:
        """Server-side ``confirm_send_message()`` tool handler."""
        await self._finish(params, CONFIRM_TOOL_NAME, await self.send_draft())

    async def _finish(self, params: FunctionCallParams, tool: str, result: dict[str, Any]) -> None:
        """Return ``result`` to the LLM, or speak it from the announcer without running the LLM."""
        speech = self.announcer(tool, result) if self.announcer else None
        if speech is None:
            await params.result_callback(result)
            return
        result = {key: value for key, value in result.items() if key != "next"}
        await params.result_callback(
            {**result, "spoken": speech}, properties=FunctionCallResultProperties(run_llm=False)
        )
        # Spoken the way an LLM reply is, so TTS and the assistant context handle it as usual.
        await params.llm.push_frame(LLMFullResponseStartFrame())
        await params.llm.push_frame(LLMTextFrame(speech))
        await params.llm.push_frame(LLMFullResponseEndFrame())

    # ------------------------------------------------------- shared steps
    # Used by the LLM tools above and by the intent engine (``messaging_intents``).
    # Errors carry a ``code`` the intent engine turns into a spoken sentence.
    @property
    def draft(self) -> Draft | None:
        """The message waiting for confirmation, if any (expired drafts are dropped)."""
        if self._draft is not None and time.monotonic() - self._draft.created_at > DRAFT_TTL_SECS:
            self._draft = None
        return self._draft

    def _person_row(self) -> dict[str, Any] | None:
        person = self._current_person()
        return person.person if person else None

    def allowed_contacts(self) -> list[Contact]:
        """The contacts the person talking may use."""
        person = self._person_row()
        return [c for c in self._gateway.contacts.all() if c.allowed_for(person)]

    def resolve(self, query: str) -> Contact | dict[str, Any]:
        """The one contact named ``query`` (a name, alias or Matrix id), or an error."""
        allowed = self.allowed_contacts()
        names = [c.name for c in allowed]
        if not allowed:
            return {
                "error": "There are no contacts you can use with this person. Contacts are set up by an admin.",
                "code": "no_contacts",
            }
        matches = match_contacts(query, allowed)
        if len(matches) == 1:
            return matches[0]
        if matches:
            return {
                "error": "Several contacts match; ask which one.",
                "code": "ambiguous",
                "candidates": [c.name for c in matches],
            }
        hidden = match_contacts(query, self._gateway.contacts.all())
        if hidden and self._person_row() is None:
            return {
                "error": "That contact belongs to a specific person and you do not know for sure who is talking. "
                "Ask them to confirm who they are.",
                "code": "unknown_person",
            }
        return {"error": f"No contact named {query!r}.", "code": "no_contact", "contacts": names}

    async def unread(self) -> list[tuple[Contact, int]]:
        """Unread message count per allowed contact (contacts without any are left out)."""
        allowed = self.allowed_contacts()
        counts = await asyncio.to_thread(self._gateway.inbox.unread_counts, [c.matrix_id for c in allowed])
        return [(c, counts[c.matrix_id]) for c in allowed if counts.get(c.matrix_id)]

    async def fetch(
        self, contacts: list[Contact], *, include_read: bool = False
    ) -> tuple[list[tuple[InboxMessage, Contact]], bool]:
        """Up to ``MAX_READ_MESSAGES`` messages from ``contacts``, marked read, and whether more are waiting."""
        by_id = {c.matrix_id: c for c in contacts}
        messages = await asyncio.to_thread(
            self._gateway.inbox.messages, list(by_id), unread_only=not include_read, limit=MAX_READ_MESSAGES + 1
        )
        more = len(messages) > MAX_READ_MESSAGES
        messages = messages[:MAX_READ_MESSAGES]
        await self._gateway.mark_read([m for m in messages if m.read_at is None])
        return [(m, by_id[m.sender]) for m in messages], more

    def prepare(self, contact: Contact, text: str) -> dict[str, Any]:
        """Make ``text`` the draft for ``contact``, replacing any earlier draft."""
        text = " ".join(text.split())
        if not text:
            return {"error": "Ask what the message should say.", "code": "no_text"}
        if len(text) > MAX_MESSAGE_LEN:
            return {
                "error": f"The message is too long; keep it under {MAX_MESSAGE_LEN} characters.",
                "code": "too_long",
            }
        person = self._person_row()
        self._draft = Draft(
            contact=contact,
            text=text,
            sender_name=person["name"] if person else None,
            person_id=person["id"] if person else None,
            user_turn=self._user_turn(),
            created_at=time.monotonic(),
        )
        return {
            "status": "draft, not sent",
            "to": contact.name,
            "text": text,
            "next": "Read the recipient and the message back and ask the user to confirm before sending.",
        }

    def cancel_draft(self) -> None:
        """Drop the draft."""
        self._draft = None

    async def send_draft(self, *, require_new_turn: bool = True) -> dict[str, Any]:
        """Send the draft, once confirmed by the person who asked for it.

        ``require_new_turn`` refuses a confirmation in the turn the draft was made
        in (the LLM confirming its own draft). The intent engine passes False: its
        confirmation is the answer to the question it asked, so always a later turn.
        """
        draft = self.draft
        if draft is None:
            return {
                "error": "There is no message waiting to be sent. Prepare it with send_message first.",
                "code": "no_draft",
            }
        if require_new_turn and self._user_turn() <= draft.user_turn:
            return {
                "error": "Not sent: read the draft back and wait for the user to confirm it first.",
                "code": "not_confirmed",
            }
        person = self._person_row()
        if not draft.contact.allowed_for(person) or (draft.person_id and (person or {}).get("id") != draft.person_id):
            return {
                "error": "Not sent: only the person who asked for this message can confirm it.",
                "code": "not_allowed",
            }
        # Shielded: an interruption must not leave a message sent but still drafted (sent twice later).
        return await asyncio.shield(self._send(draft))

    async def _send(self, draft: Draft) -> dict[str, Any]:
        body = f"{draft.sender_name}: {draft.text}" if draft.sender_name else draft.text
        try:
            await self._gateway.send_text(draft.contact, body)
        except MatrixError as exc:
            logger.warning(f"Matrix: sending to {draft.contact.name} failed: {exc}")
            return {
                "error": "The message could not be sent; the draft is kept, so you can try again later.",
                "code": "send_failed",
            }
        if self._draft is draft:
            self._draft = None
        logger.info(f"Matrix: sent a message to {draft.contact.name} for {draft.sender_name or 'a guest'}")
        return {"status": "sent", "to": draft.contact.name}

    # ------------------------------------------------------------ LLM tools
    async def _check(self) -> dict[str, Any]:
        try:
            counts = await self.unread()
        except Exception as exc:
            logger.opt(exception=exc).warning("Matrix: could not count unread messages")
            return {"error": "The messages could not be checked."}
        unread = [{"from": c.name, "unread": n} for c, n in counts]
        return {"unread": unread} if unread else {"status": "no new messages"}

    async def _read(self, query: str, include_read: bool) -> dict[str, Any]:
        if query:
            resolved = self.resolve(query)
            if isinstance(resolved, dict):
                return resolved
            contacts = [resolved]
        else:
            contacts = self.allowed_contacts()
        try:
            messages, more = await self.fetch(contacts, include_read=include_read)
        except Exception as exc:
            logger.opt(exception=exc).warning("Matrix: could not read messages")
            return {"error": "The messages could not be read."}
        if not messages:
            return {"status": "no new messages" if not include_read else "no messages"}
        result: dict[str, Any] = {"messages": [describe_message(m, c.name) for m, c in messages]}
        if more:
            result["more"] = "There are more messages; offer to read the next ones."
        return result

    def _prepare(self, query: str, text: str) -> dict[str, Any]:
        if not query:
            return {"error": "Ask who the message is for."}
        if not text:
            return {"error": "Ask what the message should say."}
        resolved = self.resolve(query)
        if isinstance(resolved, dict):
            return resolved
        return self.prepare(resolved, text)

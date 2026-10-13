# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Server-side Matrix messaging tools: check, read, draft and confirmed send."""

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace

import pytest
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from examples.shared import messaging_tools
from examples.shared.messaging_tools import (
    MESSAGING_TOOL_NAMES,
    MessagingTools,
    spoken_time,
    with_messaging_tools,
)
from messaging.config import MatrixConfig
from messaging.gateway import MatrixError, MatrixGateway
from messaging.inbox import KIND_AUDIO, KIND_TEXT, InboxMessage

BOB = "@bob:matrix.org"
CAROL = "@carol:matrix.org"
CONTACTS = f"""
contacts:
  - name: Bob
    matrix_id: "{BOB}"
  - name: Carol
    matrix_id: "{CAROL}"
    owner: Alice
"""
ALICE = SimpleNamespace(person={"id": "p-alice", "name": "Alice"})
DAN = SimpleNamespace(person={"id": "p-dan", "name": "Dan"})


class FakeGateway(MatrixGateway):
    """A gateway whose sends are recorded instead of going to Matrix."""

    def __init__(self, config):
        super().__init__(config, client=SimpleNamespace(rooms={}, close=None))
        self.sent = []
        self.fail = False
        self.read = []

    async def send_text(self, contact, text):
        if self.fail:
            raise MatrixError("offline")
        self.sent.append((contact.name, text))
        return "$e"

    async def mark_read(self, messages):
        self.read.extend(m.event_id for m in messages)
        await asyncio.to_thread(self.inbox.mark_read, [m.event_id for m in messages])


class Session:
    def __init__(self, gateway, person=None):
        self.person = person
        self.turn = 1
        self.tools = MessagingTools(gateway=gateway, current_person=lambda: self.person, user_turn=lambda: self.turn)

    def call(self, name, **arguments):
        results = []

        async def callback(result, **_):
            results.append(result)

        asyncio.run(getattr(self.tools, name)(SimpleNamespace(arguments=arguments, result_callback=callback)))
        return results[0]


@pytest.fixture
def gateway(tmp_path):
    contacts = tmp_path / "contacts.yaml"
    contacts.write_text(CONTACTS, encoding="utf-8")
    config = MatrixConfig("https://matrix.org", "@agent:matrix.org", "t", "", tmp_path, contacts)
    gw = FakeGateway(config)
    yield gw
    gw.inbox.close()


def _receive(gateway, event_id, sender, body="hi", kind=KIND_TEXT, sent_at=None):
    gateway.inbox.add(
        InboxMessage(
            event_id=event_id,
            room_id="!dm:matrix.org",
            sender=sender,
            kind=kind,
            body=body,
            sent_at=sent_at or time.time(),
        )
    )


def test_tools_schema_merges_and_replaces_same_names():
    schema = with_messaging_tools(None)
    assert [t.name for t in schema.standard_tools] == list(MESSAGING_TOOL_NAMES)
    merged = with_messaging_tools(ToolsSchema(standard_tools=schema.standard_tools))
    assert [t.name for t in merged.standard_tools] == list(MESSAGING_TOOL_NAMES)


def test_spoken_time():
    now = datetime(2026, 10, 13, 18, 0)
    assert spoken_time(datetime(2026, 10, 13, 14, 5).timestamp(), now) == "today 14:05"
    assert spoken_time(datetime(2026, 10, 12, 9, 30).timestamp(), now) == "yesterday 09:30"
    assert spoken_time(datetime(2026, 10, 5, 9, 30).timestamp(), now) == "Monday 5 October"


def test_check_and_read_mark_messages_read(gateway):
    _receive(gateway, "$1", BOB, "Dinner at eight?", sent_at=1000)
    _receive(gateway, "$2", BOB, "voice.ogg", kind=KIND_AUDIO, sent_at=2000)
    session = Session(gateway)
    assert session.call("check_messages") == {"unread": [{"from": "Bob", "unread": 2}]}

    result = session.call("read_messages")
    assert [m.get("text") for m in result["messages"]] == ["Dinner at eight?", None]
    assert result["messages"][1]["voice_message"] is True
    assert gateway.read == ["$1", "$2"]
    assert session.call("check_messages") == {"status": "no new messages"}
    assert session.call("read_messages") == {"status": "no new messages"}
    again = session.call("read_messages", contact="bob", include_read=True)
    assert len(again["messages"]) == 2


def test_read_limits_and_offers_more(gateway):
    for i in range(7):
        _receive(gateway, f"${i}", BOB, f"m{i}", sent_at=1000 + i)
    result = Session(gateway).call("read_messages")
    assert [m["text"] for m in result["messages"]] == ["m0", "m1", "m2", "m3", "m4"]
    assert "more" in result


def test_owned_contact_needs_its_owner(gateway):
    _receive(gateway, "$c", CAROL, "secret")
    guest = Session(gateway)
    assert guest.call("check_messages") == {"status": "no new messages"}
    assert "confirm who they are" in guest.call("read_messages", contact="Carol")["error"]
    assert "No contact" in Session(gateway, DAN).call("send_message", contact="Carol", text="hi")["error"]

    alice = Session(gateway, ALICE)
    assert alice.call("check_messages") == {"unread": [{"from": "Carol", "unread": 1}]}
    assert alice.call("read_messages", contact="carol")["messages"][0]["text"] == "secret"


def test_send_needs_confirmation_in_a_later_turn(gateway):
    session = Session(gateway, ALICE)
    draft = session.call("send_message", contact="Bob", text="Running late")
    assert draft["status"] == "draft, not sent" and draft["to"] == "Bob"
    # The LLM cannot confirm its own draft in the same turn.
    assert "wait for the user" in session.call("confirm_send_message")["error"]
    assert gateway.sent == []

    session.turn += 1
    assert session.call("confirm_send_message") == {"status": "sent", "to": "Bob"}
    assert gateway.sent == [("Bob", "Alice: Running late")]
    assert "no message waiting" in session.call("confirm_send_message")["error"]


def test_guest_message_is_unsigned_and_redraft_replaces(gateway):
    session = Session(gateway)
    session.call("send_message", contact="Bob", text="first")
    session.call("send_message", contact="Bob", text="second")
    session.turn += 1
    session.call("confirm_send_message")
    assert gateway.sent == [("Bob", "second")]


def test_only_the_drafter_can_confirm(gateway):
    session = Session(gateway, ALICE)
    session.call("send_message", contact="Carol", text="hi")
    session.turn += 1
    session.person = None
    assert "only the person" in session.call("confirm_send_message")["error"]
    session.person = ALICE
    assert session.call("confirm_send_message")["status"] == "sent"


def test_draft_expires(gateway, monkeypatch):
    session = Session(gateway, ALICE)
    session.call("send_message", contact="Bob", text="hi")
    session.turn += 1
    monkeypatch.setattr(messaging_tools, "DRAFT_TTL_SECS", -1)
    assert "no message waiting" in session.call("confirm_send_message")["error"]


def test_failed_send_keeps_draft(gateway):
    session = Session(gateway, ALICE)
    session.call("send_message", contact="Bob", text="hi")
    session.turn += 1
    gateway.fail = True
    assert "could not be sent" in session.call("confirm_send_message")["error"]
    gateway.fail = False
    assert session.call("confirm_send_message")["status"] == "sent"


def test_send_validation(gateway):
    session = Session(gateway)
    assert "who" in session.call("send_message", contact="", text="hi")["error"]
    assert "what" in session.call("send_message", contact="Bob", text=" ")["error"]
    too_long = "x" * (messaging_tools.MAX_MESSAGE_LEN + 1)
    assert "too long" in session.call("send_message", contact="Bob", text=too_long)["error"]
    unknown = session.call("send_message", contact="Zed", text="hi")
    assert unknown["contacts"] == ["Bob"]

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Matrix gateway: sync into the inbox, invitations, direct rooms and sending."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from nio import JoinResponse, RoomCreateResponse, RoomSendError, RoomSendResponse, SyncResponse

from messaging import gateway as gateway_module
from messaging.config import MatrixConfig
from messaging.gateway import MatrixError, MatrixGateway
from messaging.inbox import KIND_AUDIO, KIND_TEXT, SYNC_TOKEN_KEY

ME = "@agent:matrix.org"
BOB = "@bob:matrix.org"
CONTACTS = f"""
contacts:
  - name: Bob
    matrix_id: "{BOB}"
"""


def _message(event_id, sender, body, ts=1_000_000, msgtype="m.text", **content):
    return {
        "type": "m.room.message",
        "event_id": event_id,
        "sender": sender,
        "origin_server_ts": ts,
        "content": {"msgtype": msgtype, "body": body, **content},
    }


def sync_response(next_batch, join=None, invite=None):
    rooms = {
        "join": {
            room_id: {"timeline": {"events": events, "limited": False, "prev_batch": "p"}}
            for room_id, events in (join or {}).items()
        },
        "invite": {
            room_id: {
                "invite_state": {
                    "events": [
                        {
                            "type": "m.room.member",
                            "state_key": ME,
                            "sender": inviter,
                            "content": {"membership": "invite"},
                        }
                    ]
                }
            }
            for room_id, inviter in (invite or {}).items()
        },
        "leave": {},
    }
    return SyncResponse.from_dict({"next_batch": next_batch, "rooms": rooms})


class FakeClient:
    """The parts of ``nio.AsyncClient`` the gateway uses."""

    def __init__(self):
        self.syncs = []
        self.sync_calls = []
        self.rooms = {}
        self.joined = []
        self.created = []
        self.sent = []
        self.receipts = []
        self.send_errors = []

    def add_room(self, room_id, users=(ME, BOB), invited=()):
        self.rooms[room_id] = SimpleNamespace(
            users={u: None for u in users}, invited_users={u: None for u in invited}, member_count=len(users)
        )

    async def sync(self, timeout=0, since=None, full_state=None):
        self.sync_calls.append({"timeout": timeout, "since": since, "full_state": full_state})
        return self.syncs.pop(0)

    async def join(self, room_id):
        self.joined.append(room_id)
        self.add_room(room_id)
        return JoinResponse(room_id)

    async def room_create(self, **kwargs):
        self.created.append(kwargs)
        room_id = f"!new{len(self.created)}:matrix.org"
        self.add_room(room_id, users=(ME,), invited=tuple(kwargs["invite"]))
        return RoomCreateResponse(room_id)

    async def room_send(self, room_id, message_type, content):
        if self.send_errors:
            return self.send_errors.pop(0)
        self.sent.append((room_id, content))
        return RoomSendResponse(f"$sent{len(self.sent)}", room_id)

    async def room_read_markers(self, room_id, fully_read_event, read_event=None):
        self.receipts.append((room_id, read_event))

    async def close(self):
        pass


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def gateway(tmp_path: Path, client):
    contacts = tmp_path / "contacts.yaml"
    contacts.write_text(CONTACTS, encoding="utf-8")
    config = MatrixConfig(
        homeserver="https://matrix.org",
        user_id=ME,
        access_token="token",
        device_id="",
        data_dir=tmp_path,
        contacts_file=contacts,
    )
    gw = MatrixGateway(config, client=client)
    yield gw
    gw.inbox.close()


def test_first_sync_does_not_import_history(gateway, client):
    client.syncs.append(sync_response("s1", join={"!dm:matrix.org": [_message("$old", BOB, "old")]}))
    asyncio.run(gateway.sync_once())
    assert client.sync_calls[0] == {"timeout": 0, "since": None, "full_state": True}
    assert gateway.inbox.get(SYNC_TOKEN_KEY) == "s1"
    assert gateway.inbox.messages([BOB]) == []


def test_sync_stores_contact_messages_only(gateway, client):
    gateway.inbox.put(SYNC_TOKEN_KEY, "s1")
    client.add_room("!dm:matrix.org")
    client.add_room("!group:matrix.org", users=(ME, BOB, "@carol:matrix.org"))
    client.syncs.append(
        sync_response(
            "s2",
            join={
                "!dm:matrix.org": [
                    _message("$1", BOB, "hello"),
                    _message("$2", "@stranger:matrix.org", "spam"),
                    _message("$3", ME, "my own echo"),
                    _message(
                        "$4",
                        BOB,
                        "voice.ogg",
                        ts=2_000_000,
                        msgtype="m.audio",
                        url="mxc://matrix.org/abc",
                        info={"mimetype": "audio/ogg", "duration": 3200},
                    ),
                ],
                "!group:matrix.org": [_message("$5", BOB, "in a group")],
            },
        )
    )
    asyncio.run(gateway.sync_once())
    assert client.sync_calls[0]["since"] == "s1"
    messages = gateway.inbox.messages([BOB])
    assert [(m.event_id, m.kind, m.body) for m in messages] == [
        ("$1", KIND_TEXT, "hello"),
        ("$4", KIND_AUDIO, "voice.ogg"),
    ]
    assert messages[1].media_url == "mxc://matrix.org/abc"
    assert messages[1].duration_ms == 3200
    assert gateway.inbox.room_for(BOB) == "!dm:matrix.org"
    assert gateway.synced.is_set()


def test_invitations_joined_only_from_contacts(gateway, client):
    client.syncs.append(sync_response("s1", invite={"!a:matrix.org": BOB, "!b:matrix.org": "@stranger:matrix.org"}))
    asyncio.run(gateway.sync_once())
    assert client.joined == ["!a:matrix.org"]
    assert gateway.inbox.room_for(BOB) == "!a:matrix.org"


def test_sync_error_raises(gateway, client):
    client.syncs.append(SimpleNamespace(message="M_UNKNOWN_TOKEN"))
    with pytest.raises(MatrixError):
        asyncio.run(gateway.sync_once())


def test_send_creates_room_once_and_reuses_it(gateway, client):
    gateway.synced.set()
    contact = gateway.contacts.by_matrix_id(BOB)

    async def send_twice():
        await gateway.send_text(contact, "one")
        await gateway.send_text(contact, "two")

    asyncio.run(send_twice())
    assert len(client.created) == 1
    assert client.created[0]["is_direct"] and client.created[0]["invite"] == [BOB]
    assert [content["body"] for _, content in client.sent] == ["one", "two"]
    assert {room for room, _ in client.sent} == {"!new1:matrix.org"}


def test_send_replaces_room_the_contact_left(gateway, client):
    gateway.synced.set()
    client.add_room("!old:matrix.org", users=(ME,))
    gateway.inbox.set_room(BOB, "!old:matrix.org")
    asyncio.run(gateway.send_text(gateway.contacts.by_matrix_id(BOB), "hi"))
    assert client.sent[0][0] == "!new1:matrix.org"


def test_send_waits_out_rate_limits_then_fails(gateway, client, monkeypatch):
    gateway.synced.set()
    client.add_room("!dm:matrix.org")
    gateway.inbox.set_room(BOB, "!dm:matrix.org")
    contact = gateway.contacts.by_matrix_id(BOB)
    client.send_errors = [RoomSendError("slow down", "M_LIMIT_EXCEEDED", retry_after_ms=1)]
    asyncio.run(gateway.send_text(contact, "hi"))
    assert client.sent == [("!dm:matrix.org", {"msgtype": "m.text", "body": "hi"})]

    client.send_errors = [RoomSendError("forbidden", "M_FORBIDDEN")]
    with pytest.raises(MatrixError):
        asyncio.run(gateway.send_text(contact, "hi"))


def test_send_before_first_sync_fails(gateway, monkeypatch):
    monkeypatch.setattr(gateway_module, "FIRST_SYNC_WAIT_SECS", 0.01)
    with pytest.raises(MatrixError):
        asyncio.run(gateway.send_text(gateway.contacts.by_matrix_id(BOB), "hi"))


def test_mark_read_sends_one_receipt_per_room(gateway, client):
    gateway.inbox.put(SYNC_TOKEN_KEY, "s1")
    client.add_room("!dm:matrix.org")
    client.syncs.append(
        sync_response(
            "s2", join={"!dm:matrix.org": [_message("$1", BOB, "a", ts=1000), _message("$2", BOB, "b", ts=2000)]}
        )
    )
    asyncio.run(gateway.sync_once())
    asyncio.run(gateway.mark_read(gateway.inbox.messages([BOB])))
    assert client.receipts == [("!dm:matrix.org", "$2")]
    assert gateway.inbox.messages([BOB]) == []
    assert [m.event_id for m in gateway.inbox.messages([BOB], unread_only=False)] == ["$1", "$2"]

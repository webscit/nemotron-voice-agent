# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Matrix events beyond text and voice: media placeholders, edits, replies, reactions and redactions."""

import asyncio
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from loguru import logger
from nio import Event, RoomSendResponse, SyncResponse
from nio.crypto import encrypt_attachment

from examples.shared.intent_engine.responses import render_local
from examples.shared.messaging_intents import message_slots
from examples.shared.messaging_tools import describe_message
from messaging.config import MatrixConfig
from messaging.contacts import Contact
from messaging.gateway import MatrixGateway, inbox_message, short_quote, split_reply_fallback
from messaging.inbox import (
    KIND_FILE,
    KIND_IMAGE,
    KIND_LOCATION,
    KIND_STICKER,
    KIND_TEXT,
    KIND_VIDEO,
    SYNC_TOKEN_KEY,
    Inbox,
    InboxMessage,
)

ME = "@agent:matrix.org"
BOB = "@bob:matrix.org"
ROOM = "!dm:matrix.org"
CONTACTS = f"""
contacts:
  - name: Bob
    matrix_id: "{BOB}"
"""
SENTENCES = Path(__file__).resolve().parents[2] / "src/examples/shared/intent_engine/sentences"


# ------------------------------------------------------------- fixtures
def _event(event_id, content, sender=BOB, event_type="m.room.message", ts=1_000_000, **extra):
    return {
        "type": event_type,
        "event_id": event_id,
        "sender": sender,
        "origin_server_ts": ts,
        "content": content,
        **extra,
    }


def _encrypted_file(data=b"\x89PNG....", mimetype="image/png"):
    _, keys = encrypt_attachment(data)
    return {"url": "mxc://matrix.org/enc", "mimetype": mimetype, **keys}


class FakeClient:
    def __init__(self):
        self.syncs = []
        self.rooms = {ROOM: SimpleNamespace(users={ME: None, BOB: None}, invited_users={}, member_count=2)}
        self.sent = []
        self.downloads = []

    async def sync(self, timeout=0, since=None, full_state=None):
        return self.syncs.pop(0)

    async def room_send(self, room_id, message_type, content, ignore_unverified_devices=False):
        self.sent.append((room_id, content))
        return RoomSendResponse(f"$sent{len(self.sent)}", room_id)

    async def download(self, **kwargs):
        self.downloads.append(kwargs)
        raise AssertionError("media must not be downloaded")

    async def close(self):
        pass


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def gateway(tmp_path, client):
    contacts = tmp_path / "contacts.yaml"
    contacts.write_text(CONTACTS, encoding="utf-8")
    config = MatrixConfig("https://matrix.org", ME, "token", "", tmp_path, contacts)
    gw = MatrixGateway(config, client=client)
    gw.inbox.put(SYNC_TOKEN_KEY, "s0")
    gw.inbox.set_room(BOB, ROOM)
    gw.synced.set()
    gw.heard = []
    gw.add_listener(lambda message, contact: gw.heard.append(message.event_id))
    yield gw
    gw.inbox.close()


@pytest.fixture
def debug_logs():
    lines = []
    sink = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    yield lines
    logger.remove(sink)


def sync(gateway, client, *events):
    """One sync round delivering ``events`` (raw dicts) in the direct room."""
    rooms = {
        "join": {ROOM: {"timeline": {"events": list(events), "limited": False, "prev_batch": "p"}}},
        "invite": {},
        "leave": {},
    }
    client.syncs.append(SyncResponse.from_dict({"next_batch": f"s{time.monotonic_ns()}", "rooms": rooms}))
    asyncio.run(gateway.sync_once())


def decrypted(gateway, *events):
    """Events as nio hands them over once decrypted (encrypted rooms)."""
    for event in events:
        asyncio.run(gateway._on_event(ROOM, Event.parse_decrypted_event({**event, "room_id": ROOM})))


def stored(gateway):
    return gateway.inbox.messages([BOB], unread_only=False, limit=50)


# --------------------------------------------------------- placeholders
@pytest.mark.parametrize(
    ("msgtype", "kind", "content", "caption", "filename"),
    [
        ("m.image", KIND_IMAGE, {"body": "IMG_2024.jpg"}, "", "IMG_2024.jpg"),
        ("m.image", KIND_IMAGE, {"body": "At the beach", "filename": "IMG_2024.jpg"}, "At the beach", "IMG_2024.jpg"),
        ("m.file", KIND_FILE, {"body": "invoice.pdf", "filename": "invoice.pdf"}, "", "invoice.pdf"),
        ("m.video", KIND_VIDEO, {"body": "Look at this", "filename": "clip.mp4"}, "Look at this", "clip.mp4"),
    ],
)
@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_media_is_stored_as_a_placeholder(gateway, client, msgtype, kind, content, caption, filename, encrypted):
    media = {"file": _encrypted_file()} if encrypted else {"url": "mxc://matrix.org/plain"}
    event = _event("$m", {"msgtype": msgtype, **content, "info": {"mimetype": "image/png", "size": 9}, **media})
    if encrypted:
        decrypted(gateway, event)
    else:
        sync(gateway, client, event)
    [message] = stored(gateway)
    assert (message.kind, message.body, message.filename) == (kind, caption, filename)
    assert message.media_url == ("mxc://matrix.org/enc" if encrypted else "mxc://matrix.org/plain")
    assert bool(message.media_key) is encrypted
    assert message.read_at is None
    # A new message for the announcer; never downloaded.
    assert gateway.heard == ["$m"]
    assert client.downloads == []


def test_encrypted_media_without_a_usable_key_is_still_announced(gateway):
    event = _event("$m", {"msgtype": "m.image", "body": "photo.jpg", "file": {"url": "mxc://matrix.org/enc"}})
    decrypted(gateway, event)
    [message] = stored(gateway)
    assert (message.kind, message.media_key) == (KIND_IMAGE, None)


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_location_never_keeps_the_geo_uri(gateway, client, encrypted):
    events = [
        _event(
            "$l1", {"msgtype": "m.location", "body": "Location geo:48.85,2.35 at 12:00", "geo_uri": "geo:48.85,2.35"}
        ),
        _event(
            "$l2",
            {
                "msgtype": "m.location",
                "body": "geo:48.85,2.35",
                "geo_uri": "geo:48.85,2.35",
                "org.matrix.msc3488.location": {"uri": "geo:48.85,2.35", "description": "Café de Flore"},
            },
            ts=2_000_000,
        ),
    ]
    if encrypted:
        decrypted(gateway, *events)
    else:
        sync(gateway, client, *events)
    assert [(m.kind, m.body) for m in stored(gateway)] == [(KIND_LOCATION, ""), (KIND_LOCATION, "Café de Flore")]


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_sticker_is_stored_as_a_placeholder(gateway, client, encrypted):
    media = {"file": _encrypted_file()} if encrypted else {"url": "mxc://matrix.org/sticker"}
    event = _event("$s", {"body": "Happy cat", "info": {"mimetype": "image/webp"}, **media}, event_type="m.sticker")
    if encrypted:
        # nio's sticker schema wants a plain url: the encrypted one comes out as a BadEvent.
        decrypted(gateway, event)
    else:
        sync(gateway, client, event)
    [message] = stored(gateway)
    assert (message.kind, message.body, message.filename) == (KIND_STICKER, "Happy cat", None)
    assert gateway.heard == ["$s"]


def test_media_from_strangers_is_ignored(gateway, client):
    sync(gateway, client, _event("$m", {"msgtype": "m.image", "body": "x.jpg", "url": "mxc://a/b"}, sender="@x:y.org"))
    assert stored(gateway) == []


# ----------------------------------------------------------------- edits
def _edit(event_id, target, body, sender=BOB, **new):
    return _event(
        event_id,
        {
            "msgtype": "m.text",
            "body": f"* {body}",
            "m.new_content": {"msgtype": "m.text", "body": body, **new},
            "m.relates_to": {"rel_type": "m.replace", "event_id": target},
        },
        sender=sender,
        ts=3_000_000,
    )


@pytest.mark.parametrize("encrypted", [False, True], ids=["plain", "encrypted"])
def test_edit_replaces_the_body_without_a_new_arrival(gateway, client, encrypted):
    sync(gateway, client, _event("$1", {"msgtype": "m.text", "body": "See you at 5"}))
    deliver = (lambda *e: decrypted(gateway, *e)) if encrypted else (lambda *e: sync(gateway, client, *e))
    deliver(_edit("$2", "$1", "See you at 6"))
    [message] = stored(gateway)
    assert (message.event_id, message.body, message.read_at) == ("$1", "See you at 6", None)
    assert gateway.heard == ["$1"]


def test_edit_keeps_the_read_state(gateway, client):
    sync(gateway, client, _event("$1", {"msgtype": "m.text", "body": "See you at 5"}))
    gateway.inbox.mark_read(["$1"])
    sync(gateway, client, _edit("$2", "$1", "See you at 6"))
    [message] = stored(gateway)
    assert message.body == "See you at 6"
    assert message.read_at is not None
    assert gateway.inbox.unread_counts([BOB]) == {}


def test_edit_of_a_caption(gateway, client):
    sync(gateway, client, _event("$1", {"msgtype": "m.image", "body": "IMG.jpg", "url": "mxc://a/b"}))
    edit = _edit("$2", "$1", "Our new car", msgtype="m.image", filename="IMG.jpg")
    edit["content"]["m.new_content"]["msgtype"] = "m.image"
    sync(gateway, client, edit)
    [message] = stored(gateway)
    assert (message.kind, message.body, message.filename) == (KIND_IMAGE, "Our new car", "IMG.jpg")


def test_edit_of_an_unknown_event_is_ignored(gateway, client, debug_logs):
    sync(gateway, client, _edit("$2", "$unknown", "Hello"))
    assert stored(gateway) == []
    assert gateway.heard == []
    assert [line for line in debug_logs if "$unknown" in line and "not in the inbox" in line]


def test_edit_by_somebody_else_is_ignored(gateway, client):
    gateway.contacts = SimpleNamespace(
        by_matrix_id=lambda matrix_id: Contact(name="X", matrix_id=matrix_id), all=lambda: []
    )
    sync(gateway, client, _event("$1", {"msgtype": "m.text", "body": "Original"}))
    sync(gateway, client, _edit("$2", "$1", "Forged", sender="@carol:matrix.org"))
    assert [m.body for m in stored(gateway)] == ["Original"]


# --------------------------------------------------------------- replies
def _reply(event_id, target, body, quote=None, quoted_sender=ME):
    fallback = f"> <{quoted_sender}> {quote}\n\n" if quote is not None else ""
    return _event(
        event_id,
        {"msgtype": "m.text", "body": f"{fallback}{body}", "m.relates_to": {"m.in_reply_to": {"event_id": target}}},
        ts=4_000_000,
    )


def test_split_reply_fallback():
    body = "> <@agent:matrix.org> Alice: Are you coming\n> for dinner?\n\nYes, at 8"
    assert split_reply_fallback(body) == ("Yes, at 8", "Alice: Are you coming for dinner?", ME)
    assert split_reply_fallback("No fallback") == ("No fallback", None, None)
    long = " ".join(["word"] * 40)
    assert short_quote(long).endswith("…") and len(short_quote(long)) <= 81


def test_reply_to_the_agent_quotes_the_sent_message(gateway, client):
    bob = gateway.contacts.by_matrix_id(BOB)
    event_id = asyncio.run(gateway.send_text(bob, "Alice: Are you coming for dinner?"))
    # Modern clients send no fallback: the quote comes from what the agent sent.
    sync(gateway, client, _reply("$r", event_id, "Yes, at 8"))
    [message] = stored(gateway)
    assert (message.body, message.reply_to, message.reply_quote, message.reply_mine) == (
        "Yes, at 8",
        event_id,
        "Alice: Are you coming for dinner?",
        True,
    )


def test_reply_fallback_is_stripped_and_quoted(gateway, client):
    sync(gateway, client, _reply("$r", "$gone", "Yes, at 8", quote="Are you coming?"))
    [message] = stored(gateway)
    assert (message.body, message.reply_quote, message.reply_mine) == ("Yes, at 8", "Are you coming?", True)


def test_reply_to_a_stored_message(gateway, client):
    sync(gateway, client, _event("$1", {"msgtype": "m.text", "body": "I'll be late"}))
    sync(gateway, client, _reply("$r", "$1", "Very late, actually", quote="I'll be late", quoted_sender=BOB))
    message = stored(gateway)[-1]
    assert (message.body, message.reply_quote, message.reply_mine) == ("Very late, actually", "I'll be late", False)


def test_reply_to_an_unknown_message_without_fallback(gateway, client):
    sync(gateway, client, _reply("$r", "$gone", "Yes"))
    [message] = stored(gateway)
    assert (message.body, message.reply_to, message.reply_quote) == ("Yes", "$gone", None)


def test_thread_fallback_is_not_a_reply(gateway, client):
    event = _event(
        "$t",
        {
            "msgtype": "m.text",
            "body": "In a thread",
            "m.relates_to": {
                "rel_type": "m.thread",
                "event_id": "$root",
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": "$root"},
            },
        },
    )
    sync(gateway, client, event)
    [message] = stored(gateway)
    assert (message.body, message.reply_to) == ("In a thread", None)


# ------------------------------------------------- reactions, redactions
def test_reaction_is_ignored(gateway, client):
    sync(gateway, client, _event("$1", {"msgtype": "m.text", "body": "Hi"}))
    reaction = _event(
        "$x", {"m.relates_to": {"rel_type": "m.annotation", "event_id": "$1", "key": "👍"}}, event_type="m.reaction"
    )
    sync(gateway, client, reaction)
    decrypted(gateway, {**reaction, "event_id": "$y"})
    assert [(m.event_id, m.body) for m in stored(gateway)] == [("$1", "Hi")]
    assert gateway.heard == ["$1"]


@pytest.mark.parametrize("v11", [False, True], ids=["redacts", "content.redacts"])
def test_redaction_removes_the_message(gateway, client, v11):
    sync(
        gateway,
        client,
        _event("$1", {"msgtype": "m.text", "body": "Oops"}),
        _event("$2", {"msgtype": "m.text", "body": "Hi"}),
    )
    redaction = (
        _event("$x", {"redacts": "$1"}, event_type="m.room.redaction")
        if v11
        else _event("$x", {}, event_type="m.room.redaction", redacts="$1")
    )
    sync(gateway, client, redaction)
    assert [m.event_id for m in stored(gateway)] == ["$2"]
    assert gateway.heard == ["$1", "$2"]
    # Waiting for an announcement: no longer unread, so the announcer drops it.
    assert gateway.inbox.unread_among(["$1", "$2"]) == {"$2"}


def test_redaction_of_an_unknown_event_is_ignored(gateway, client, debug_logs):
    sync(gateway, client, _event("$1", {"msgtype": "m.text", "body": "Hi"}))
    sync(gateway, client, _event("$x", {}, event_type="m.room.redaction", redacts="$unknown"))
    assert [m.event_id for m in stored(gateway)] == ["$1"]
    assert [line for line in debug_logs if "$unknown" in line and "not in the inbox" in line]


def test_redaction_by_somebody_else_is_ignored(gateway, client):
    gateway.contacts = SimpleNamespace(
        by_matrix_id=lambda matrix_id: Contact(name="X", matrix_id=matrix_id), all=lambda: []
    )
    sync(gateway, client, _event("$1", {"msgtype": "m.text", "body": "Hi"}))
    sync(gateway, client, _event("$x", {}, event_type="m.room.redaction", redacts="$1", sender="@carol:matrix.org"))
    assert [m.event_id for m in stored(gateway)] == ["$1"]


def test_already_redacted_event_is_not_stored():
    event = Event.parse_event(
        _event("$1", {}, unsigned={"redacted_because": {"type": "m.room.redaction", "sender": BOB}})
    )
    assert inbox_message(ROOM, event) is None


# ------------------------------------------------------------- migration
def test_inbox_rows_from_before_media_keep_working(tmp_path):
    path = tmp_path / "inbox.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE messages (
            event_id TEXT PRIMARY KEY, room_id TEXT NOT NULL, sender TEXT NOT NULL, kind TEXT NOT NULL,
            body TEXT NOT NULL, media_url TEXT, mimetype TEXT, duration_ms INTEGER, media_key TEXT,
            transcript TEXT, sent_at REAL NOT NULL, received_at REAL NOT NULL, read_at REAL
        );
        INSERT INTO messages (event_id, room_id, sender, kind, body, sent_at, received_at)
        VALUES ('$old', '!dm:matrix.org', '@bob:matrix.org', 'text', 'Before the upgrade', 1, 1);
        """
    )
    old.commit()
    old.close()
    inbox = Inbox(path)
    [message] = inbox.messages([BOB])
    assert (message.body, message.filename, message.reply_to, message.reply_quote, message.reply_mine) == (
        "Before the upgrade",
        None,
        None,
        None,
        False,
    )
    assert inbox.edit("$old", ROOM, BOB, "Edited")
    assert inbox.message("$old").body == "Edited"
    inbox.add_sent("$s", ROOM, "Hello")
    assert inbox.sent_body("$s") == "Hello"
    inbox.close()


def test_sent_messages_are_bounded(tmp_path, monkeypatch):
    from messaging import inbox as inbox_module

    monkeypatch.setattr(inbox_module, "MAX_SENT", 3)
    inbox = Inbox(tmp_path / "inbox.db")
    for n in range(5):
        inbox.add_sent(f"$s{n}", ROOM, f"m{n}")
    assert [inbox.sent_body(f"$s{n}") for n in range(5)] == [None, None, "m2", "m3", "m4"]
    inbox.close()


# ------------------------------------------------------------- rendering
def _template(language):
    data = yaml.safe_load((SENTENCES / language / "messaging.yaml").read_text(encoding="utf-8"))
    return data["responses"]["intents"]["Messaging"]["messages"]


def _render(language, *messages):
    now = datetime.now()
    bob = Contact(name="Bob", matrix_id=BOB)
    slots = {"messages": [message_slots(m, bob, now) for m in messages], "more": False, "reply_to": None}
    return render_local(_template(language), slots)


def _msg(kind=KIND_TEXT, body="", **fields):
    return InboxMessage(event_id="$1", room_id=ROOM, sender=BOB, kind=kind, body=body, sent_at=time.time(), **fields)


@pytest.mark.parametrize(
    ("message", "en", "fr"),
    [
        (
            _msg(KIND_IMAGE, "At the beach", filename="IMG.jpg"),
            "Bob sent you a photo: At the beach.",
            "Bob t'a envoyé une photo : At the beach.",
        ),
        (_msg(KIND_IMAGE, filename="IMG.jpg"), "Bob sent you a photo.", "Bob t'a envoyé une photo."),
        (
            _msg(KIND_FILE, filename="invoice.pdf"),
            "Bob sent you a file named invoice.pdf.",
            "Bob t'a envoyé un fichier nommé invoice.pdf.",
        ),
        (_msg(KIND_VIDEO, "Look!"), "Bob sent you a video: Look!", "Bob t'a envoyé une vidéo : Look!"),
        (
            _msg(KIND_LOCATION, "Café de Flore"),
            "Bob sent you a location: Café de Flore.",
            "Bob t'a envoyé une position : Café de Flore.",
        ),
        (
            _msg(KIND_STICKER, "Happy cat"),
            "Bob sent you a sticker: Happy cat.",
            "Bob t'a envoyé un autocollant : Happy cat.",
        ),
        (
            _msg(body="Yes, at 8", reply_to="$0", reply_quote="Are you coming?", reply_mine=True),
            'Bob replied to your message "Are you coming?": Yes, at 8.',
            "Bob a répondu à ton message « Are you coming? » : Yes, at 8.",
        ),
        (
            _msg(body="Very late", reply_to="$0", reply_quote="I'll be late"),
            'Bob replied to the message "I\'ll be late": Very late.',
            "Bob a répondu au message « I'll be late » : Very late.",
        ),
        (
            _msg(body="Yes", reply_to="$0"),
            "Bob replied to an earlier message: Yes.",
            "Bob a répondu à un message précédent : Yes.",
        ),
        (
            _msg(body="Yes", reply_to="$0", reply_mine=True),
            "Bob replied to one of your messages: Yes.",
            "Bob a répondu à un de tes messages : Yes.",
        ),
        (_msg(body="Hello"), "Bob wrote: Hello.", "Bob a écrit : Hello."),
    ],
)
def test_messages_are_said_in_english_and_french(message, en, fr):
    assert en in _render("en", message)
    assert fr in _render("fr", message)


def test_llm_tool_result_describes_media_and_replies():
    photo = describe_message(_msg(KIND_IMAGE, "At the beach", filename="IMG.jpg"), "Bob")
    assert photo["attachment"] == {"type": "photo", "caption": "At the beach"}
    assert "cannot see" in photo["note"] and "text" not in photo
    file = describe_message(_msg(KIND_FILE, filename="invoice.pdf"), "Bob")
    assert file["attachment"] == {"type": "file", "filename": "invoice.pdf"}
    reply = describe_message(_msg(body="Yes", reply_to="$0", reply_quote="Coming?", reply_mine=True), "Bob")
    assert reply["text"] == "Yes"
    assert reply["in_reply_to"] == {"quote": "Coming?", "your_message": True}
    assert "in_reply_to" not in describe_message(_msg(body="Hi"), "Bob")

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Matrix end-to-end encryption: device keys, encrypted rooms and attachments, undecryptable events."""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from loguru import logger
from nio import (
    Event,
    LocalProtocolError,
    MegolmEvent,
    RoomCreateResponse,
    RoomEncryptedAudio,
    RoomMessageText,
    UploadResponse,
    WhoamiResponse,
)
from nio.crypto import encrypt_attachment
from nio.exceptions import EncryptionError
from test_matrix_gateway import BOB, CONTACTS, ME, FakeClient, _message, sync_response

from messaging import gateway as gateway_module
from messaging.config import MatrixConfig
from messaging.gateway import MatrixError, MatrixGateway, decrypt_media, inbox_message
from messaging.inbox import KIND_AUDIO, SYNC_TOKEN_KEY, Inbox, InboxMessage


def _config(tmp_path, encryption=True):
    contacts = tmp_path / "contacts.yaml"
    contacts.write_text(CONTACTS, encoding="utf-8")
    return MatrixConfig("https://matrix.org", ME, "token", "", tmp_path, contacts, encryption=encryption)


def _encrypted(event_id, sender=BOB):
    return {
        "type": "m.room.encrypted",
        "event_id": event_id,
        "sender": sender,
        "origin_server_ts": 1_000_000,
        "content": {
            "algorithm": "m.megolm.v1.aes-sha2",
            "ciphertext": "AwgA",
            "sender_key": "senderkey",
            "device_id": "BOBPHONE",
            "session_id": "session1",
        },
    }


@pytest.fixture
def logs():
    lines = []
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    yield lines
    logger.remove(sink)


# ------------------------------------------------------- real nio client
def _real_gateway(tmp_path, server_devices=()):
    """A gateway with a real nio client; whoami and the keys query answered locally."""
    gateway = MatrixGateway(_config(tmp_path))
    queries = []

    async def whoami():
        return WhoamiResponse(ME, "AGENTDEV", False)

    async def send(method, path, data=None, headers=None, **kwargs):
        queries.append((method, path.split("?")[0]))

        async def json():
            return {"device_keys": {ME: {device: {} for device in server_devices}}}

        return SimpleNamespace(status=200, json=json)

    gateway._client.whoami = whoami
    gateway._client.send = send
    gateway.key_queries = queries
    return gateway


def test_login_loads_the_device_keys_of_the_token(tmp_path, logs):
    gateway = _real_gateway(tmp_path)

    async def scenario():
        gateway.lock_store()
        assert not gateway.encrypted
        await gateway._login()
        assert gateway.encrypted
        assert gateway._client.device_id == "AGENTDEV"
        # A new device: its keys are uploaded after the first sync.
        assert gateway._client.should_upload_keys
        assert gateway.key_queries == [("POST", "/_matrix/client/v3/keys/query")]
        await gateway.close()
        # Next start: the store is there, the device's keys on the server are ours.
        again = _real_gateway(tmp_path, server_devices=["AGENTDEV"])
        again.lock_store()
        await again._login()
        assert again.encrypted
        assert again.key_queries == []
        await again.close()

    asyncio.run(scenario())
    assert (tmp_path / "crypto" / f"{ME}_AGENTDEV.db").exists()


def test_device_with_keys_of_another_client_stays_unencrypted(tmp_path, logs):
    # The token of an Element session: its device already has keys this store does not hold.
    gateway = _real_gateway(tmp_path, server_devices=["AGENTDEV"])

    async def scenario():
        gateway.lock_store()
        await gateway._login()
        assert not gateway.encrypted
        assert not gateway._login_pending
        await gateway.close()

    asyncio.run(scenario())
    assert [line for line in logs if "already has encryption keys" in line]
    assert not (tmp_path / "crypto" / f"{ME}_AGENTDEV.db").exists()


def test_decrypted_voice_message_parses_as_an_encrypted_attachment():
    # The parser nio runs on a decrypted megolm payload.
    _, keys = encrypt_attachment(b"OggS")
    event = Event.parse_decrypted_event(
        {
            "type": "m.room.message",
            "event_id": "$v",
            "sender": BOB,
            "origin_server_ts": 1,
            "room_id": "!dm:matrix.org",
            "content": {"msgtype": "m.audio", "body": "v", "file": {"url": "mxc://matrix.org/enc", **keys}},
        }
    )
    assert isinstance(event, RoomEncryptedAudio)
    assert inbox_message("!dm:matrix.org", event).media_key


def test_encryption_failure_of_a_send_is_a_matrix_error(enc_gateway, enc_client):
    enc_client.add_room("!dm:matrix.org")
    enc_gateway.inbox.set_room(BOB, "!dm:matrix.org")

    async def room_send(*args, **kwargs):
        raise LocalProtocolError("No such room")

    enc_client.room_send = room_send
    contact = enc_gateway.contacts.by_matrix_id(BOB)
    with pytest.raises(MatrixError, match="No such room"):
        asyncio.run(enc_gateway.send_text(contact, "hello"))
    with pytest.raises(MatrixError, match="No such room"):
        asyncio.run(enc_gateway.send_voice(contact, b"OggS", duration_ms=1, waveform=[1], body="x"))


def test_one_process_owns_the_key_store(tmp_path):
    async def scenario():
        first = MatrixGateway(_config(tmp_path))
        first.lock_store()
        second = MatrixGateway(_config(tmp_path))
        with pytest.raises(MatrixError, match="another process"):
            second.lock_store()
        await second.close()
        await first.close()
        third = MatrixGateway(_config(tmp_path))
        third.lock_store()
        await third.close()

    asyncio.run(scenario())


def test_start_gateway_turns_messaging_off_when_the_store_is_taken(tmp_path, monkeypatch):
    async def scenario():
        owner = MatrixGateway(_config(tmp_path))
        owner.lock_store()
        monkeypatch.setattr(gateway_module, "_gateway", None)
        assert await gateway_module.start_gateway(_config(tmp_path)) is None
        await owner.close()

    asyncio.run(scenario())


def test_encrypted_voice_message_round_trip():
    audio = b"OggS" + bytes(range(256)) * 8
    ciphertext, keys = encrypt_attachment(audio)
    event = RoomEncryptedAudio.from_dict(
        {
            "type": "m.room.message",
            "event_id": "$v",
            "sender": BOB,
            "origin_server_ts": 2_000_000,
            "content": {
                "msgtype": "m.audio",
                "body": "voice-message.ogg",
                "info": {"mimetype": "audio/ogg", "duration": 3200},
                "file": {"url": "mxc://matrix.org/enc", "mimetype": "audio/ogg", **keys},
            },
        }
    )
    message = inbox_message("!dm:matrix.org", event)
    assert (message.kind, message.media_url, message.duration_ms) == (KIND_AUDIO, "mxc://matrix.org/enc", 3200)
    assert decrypt_media(ciphertext, message.media_key) == audio
    with pytest.raises(MatrixError, match="decrypt"):
        decrypt_media(ciphertext[:-1] + b"x", message.media_key)


def test_inbox_from_before_encryption_gets_the_key_column(tmp_path):
    path = tmp_path / "inbox.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE messages (
            event_id TEXT PRIMARY KEY, room_id TEXT NOT NULL, sender TEXT NOT NULL, kind TEXT NOT NULL,
            body TEXT NOT NULL, media_url TEXT, mimetype TEXT, duration_ms INTEGER, sent_at REAL NOT NULL,
            received_at REAL NOT NULL, read_at REAL
        );
        INSERT INTO messages VALUES ('$1', '!dm', '@bob:matrix.org', 'text', 'hi', NULL, NULL, NULL, 1, 1, NULL);
        """
    )
    old.close()
    inbox = Inbox(path)
    assert [m.body for m in inbox.messages([BOB])] == ["hi"]
    inbox.add(InboxMessage("$2", "!dm", BOB, KIND_AUDIO, "v", 2, media_url="mxc://x/y", media_key='{"k": 1}'))
    assert inbox.messages([BOB])[-1].media_key == '{"k": 1}'
    inbox.close()
    # Opening it again does not add the column twice.
    Inbox(path).close()


# ------------------------------------------------------ encrypted client
class EncryptedClient(FakeClient):
    """A logged-in client with encryption: rooms it creates turn encrypted when the sync shows them."""

    def __init__(self):
        super().__init__()
        self.olm = object()
        self.uploads = []
        self.key_requests = []
        self.keys = SimpleNamespace(upload=False, query=False, claim=False)
        self.key_calls = []
        self.decryptable = set()
        self.devices = {BOB: ["BOBPHONE"]}
        self.device_store = SimpleNamespace(
            active_user_devices=lambda user: [SimpleNamespace(id=d) for d in self.devices.get(user, [])]
        )
        self.sync_shows_new_rooms_after = 0.05

    def add_room(self, room_id, users=(ME, BOB), invited=(), encrypted=True):
        super().add_room(room_id, users, invited)
        self.rooms[room_id].encrypted = encrypted

    async def room_create(self, **kwargs):
        self.created.append(kwargs)
        room_id = f"!new{len(self.created)}:matrix.org"

        async def synced():
            await asyncio.sleep(self.sync_shows_new_rooms_after)
            encrypted = any(e["type"] == "m.room.encryption" for e in kwargs.get("initial_state", []))
            self.add_room(room_id, users=(ME,), invited=tuple(kwargs["invite"]), encrypted=encrypted)

        self._room_task = asyncio.ensure_future(synced())
        return RoomCreateResponse(room_id)

    async def upload(self, data_provider, content_type, filename=None, filesize=None, encrypt=False):
        data = data_provider(0, 0).read()
        self.uploads.append((data, encrypt))
        keys = {"key": {"k": "secret"}, "iv": "iv", "hashes": {"sha256": "h"}, "v": "v2"} if encrypt else None
        return UploadResponse("mxc://matrix.org/voice1"), keys

    @property
    def should_upload_keys(self):
        return self.keys.upload

    @property
    def should_query_keys(self):
        return self.keys.query

    @property
    def should_claim_keys(self):
        return self.keys.claim

    async def send_to_device_messages(self):
        self.key_calls.append("to_device")

    async def keys_upload(self):
        self.key_calls.append("upload")
        self.keys.upload = False

    async def keys_query(self):
        self.key_calls.append("query")
        self.keys.query = False

    async def keys_claim(self, users):
        self.key_calls.append("claim")
        self.keys.claim = False

    def get_users_for_key_claiming(self):
        return {BOB: ["BOBPHONE"]}

    async def request_room_key(self, event):
        self.key_requests.append(event.event_id)

    def decrypt_event(self, event):
        if event.event_id not in self.decryptable:
            raise EncryptionError("no session")
        return RoomMessageText.from_dict(_message(event.event_id, event.sender, "déchiffré"))


@pytest.fixture
def enc_client():
    return EncryptedClient()


@pytest.fixture
def enc_gateway(tmp_path, enc_client):
    gw = MatrixGateway(_config(tmp_path), client=enc_client)
    gw.inbox.put(SYNC_TOKEN_KEY, "s1")
    gw.synced.set()
    yield gw
    gw.inbox.close()


def test_first_sync_of_the_process_loads_every_room(enc_gateway, enc_client):
    enc_client.syncs += [sync_response("s2"), sync_response("s3")]
    asyncio.run(enc_gateway.sync_once())
    asyncio.run(enc_gateway.sync_once())
    assert enc_client.sync_calls[0] == {"timeout": 0, "since": "s1", "full_state": True}
    assert enc_client.sync_calls[1] == {"timeout": gateway_module.SYNC_TIMEOUT_MS, "since": "s2", "full_state": False}


def test_keys_are_shared_after_each_sync_in_nio_order(enc_gateway, enc_client):
    enc_client.keys = SimpleNamespace(upload=True, query=True, claim=True)
    enc_client.syncs.append(sync_response("s2"))
    asyncio.run(enc_gateway.sync_once())
    assert enc_client.key_calls == ["to_device", "upload", "query", "claim"]


def test_a_new_device_of_a_contact_is_logged(enc_gateway, enc_client, logs):
    async def query():
        enc_client.keys.query = True
        await enc_gateway._manage_keys()

    asyncio.run(query())
    # The first devices seen are not news.
    assert not [line for line in logs if "new device" in line]
    enc_client.devices[BOB] = ["BOBPHONE", "BOBLAPTOP"]
    asyncio.run(query())
    assert [line for line in logs if "Bob has a new device (BOBLAPTOP)" in line]


def test_undecryptable_message_is_kept_until_its_key_arrives(enc_gateway, enc_client):
    enc_client.add_room("!dm:matrix.org")
    heard = []
    enc_gateway.add_listener(lambda message, contact: heard.append(message.body))
    enc_client.syncs.append(
        sync_response("s2", join={"!dm:matrix.org": [_encrypted("$e1"), _encrypted("$spam", "@x:matrix.org")]})
    )
    asyncio.run(enc_gateway.sync_once())
    assert isinstance(enc_gateway._pending_decrypt["$e1"][1], MegolmEvent)
    assert enc_client.key_requests == ["$e1"]
    assert enc_gateway.inbox.messages([BOB]) == []
    # The key arrived (forwarded by Bob's device): decrypted after the next sync.
    enc_client.decryptable.add("$e1")
    enc_client.syncs.append(sync_response("s3"))
    asyncio.run(enc_gateway.sync_once())
    assert [m.body for m in enc_gateway.inbox.messages([BOB])] == ["déchiffré"]
    assert heard == ["déchiffré"]
    assert not enc_gateway._pending_decrypt


def test_without_encryption_encrypted_messages_are_ignored_with_a_warning(tmp_path, logs):
    client = FakeClient()
    client.add_room("!dm:matrix.org")
    gw = MatrixGateway(_config(tmp_path, encryption=False), client=client)
    gw.inbox.put(SYNC_TOKEN_KEY, "s1")
    client.syncs += [
        sync_response("s2", join={"!dm:matrix.org": [_encrypted("$e1")]}),
        sync_response("s3", join={"!dm:matrix.org": [_encrypted("$e2")]}),
    ]
    asyncio.run(gw.sync_once())
    asyncio.run(gw.sync_once())
    assert gw.inbox.messages([BOB]) == []
    assert len([line for line in logs if "ignoring encrypted messages" in line]) == 1
    gw.inbox.close()


def test_new_room_is_encrypted_and_used_once_synced(enc_gateway, enc_client):
    contact = enc_gateway.contacts.by_matrix_id(BOB)
    asyncio.run(enc_gateway.send_text(contact, "hello"))
    (created,) = enc_client.created
    assert created["initial_state"] == [
        {"type": "m.room.encryption", "state_key": "", "content": {"algorithm": "m.megolm.v1.aes-sha2"}}
    ]
    assert enc_client.sent == [("!new1:matrix.org", {"msgtype": "m.text", "body": "hello"})]


def test_send_fails_when_the_new_room_never_syncs(enc_gateway, enc_client, monkeypatch):
    monkeypatch.setattr(gateway_module, "NEW_ROOM_WAIT_SECS", 0.1)
    enc_client.sync_shows_new_rooms_after = 10
    contact = enc_gateway.contacts.by_matrix_id(BOB)

    async def scenario():
        with pytest.raises(MatrixError, match="did not sync"):
            await enc_gateway.send_text(contact, "hello")
        enc_client._room_task.cancel()

    asyncio.run(scenario())
    assert enc_client.sent == []


def test_voice_message_to_an_encrypted_room_is_an_encrypted_attachment(enc_gateway, enc_client):
    enc_client.add_room("!dm:matrix.org")
    enc_gateway.inbox.set_room(BOB, "!dm:matrix.org")
    contact = enc_gateway.contacts.by_matrix_id(BOB)
    asyncio.run(enc_gateway.send_voice(contact, b"OggS", duration_ms=1000, waveform=[1], body="Alice: hi"))
    assert enc_client.uploads == [(b"OggS", True)]
    ((_, content),) = enc_client.sent
    assert "url" not in content
    assert content["file"]["url"] == "mxc://matrix.org/voice1"
    assert content["file"]["key"] == {"k": "secret"}
    assert content["info"]["duration"] == 1000


def test_voice_message_to_an_unencrypted_room_stays_plain(enc_gateway, enc_client):
    enc_client.add_room("!dm:matrix.org", encrypted=False)
    enc_gateway.inbox.set_room(BOB, "!dm:matrix.org")
    contact = enc_gateway.contacts.by_matrix_id(BOB)
    asyncio.run(enc_gateway.send_voice(contact, b"OggS", duration_ms=1000, waveform=[1], body="Alice: hi"))
    assert enc_client.uploads == [(b"OggS", False)]
    ((_, content),) = enc_client.sent
    assert content["url"] == "mxc://matrix.org/voice1" and "file" not in content


def test_received_encrypted_voice_is_decrypted_on_download(enc_gateway, enc_client):
    audio = b"OggS-voice"
    ciphertext, keys = encrypt_attachment(audio)

    async def download(mxc_url):
        assert mxc_url == "mxc://matrix.org/enc"
        return ciphertext

    enc_gateway.download = download
    event = RoomEncryptedAudio.from_dict(
        {
            "type": "m.room.message",
            "event_id": "$v",
            "sender": BOB,
            "origin_server_ts": 1,
            "content": {"msgtype": "m.audio", "body": "v", "file": {"url": "mxc://matrix.org/enc", **keys}},
        }
    )
    assert asyncio.run(enc_gateway.download_media(inbox_message("!dm:matrix.org", event))) == audio

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent's Matrix account: one client per server process, shared by all sessions.

The gateway syncs in the background, stores what contacts send in the inbox, and
sends messages to contacts in a direct room it creates or that the contact
invited it to. Live sessions listen for new messages (``add_listener``) to
announce them. Messages from anybody who is not in the contacts file are
ignored, and invitations from them are left pending.

End-to-end encryption (``MATRIX_ENCRYPTION``, on by default) is handled by
nio's olm machine (vodozemac): the agent's device keys live in
``<MATRIX_DATA_DIR>/crypto``, encrypted rooms are read and answered, the rooms
the agent creates are encrypted, and voice messages are encrypted attachments.
Messages are encrypted for every device of a contact (trust on first use); a
device that appears later is logged. Unencrypted rooms keep working.

Photos, files, videos, locations and stickers are stored as placeholders (their
caption or file name; the media is never downloaded). An edit replaces the text
of the message it edits and a redaction deletes it, without counting as a new
message; a reply loses its quoted fallback and keeps a short quote of the message
it answers. Reactions are ignored.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import io
import json
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import IO, Any

from loguru import logger
from nio import (
    Api,
    AsyncClient,
    AsyncClientConfig,
    DownloadError,
    ErrorResponse,
    InviteMemberEvent,
    JoinError,
    KeysUploadError,
    LocalProtocolError,
    MegolmEvent,
    RoomEncryptedAudio,
    RoomMessageAudio,
    RoomMessageEmote,
    RoomMessageText,
    RoomPreset,
    SyncResponse,
    UploadResponse,
    WhoamiResponse,
)
from nio.crypto import ENCRYPTION_ENABLED, decrypt_attachment
from nio.exceptions import EncryptionError

from messaging.config import MatrixConfig
from messaging.contacts import Contact, ContactBook
from messaging.inbox import (
    KIND_AUDIO,
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
from messaging.voice import OGG_MIMETYPE

SYNC_TIMEOUT_MS = 30_000
RETRY_MIN_SECS = 2.0
RETRY_MAX_SECS = 60.0
# Rate-limited requests are retried this many times (matrix.org limits sends per second).
MAX_RATE_LIMIT_RETRIES = 3
# How long a send waits for the first sync after the server started.
FIRST_SYNC_WAIT_SECS = 10.0
# A room with more members than this is not a direct chat: its messages are not read.
MAX_DM_MEMBERS = 2
# Received media larger than this is not downloaded (a few minutes of Opus is well under 1 MB).
MAX_MEDIA_BYTES = 10 * 1024 * 1024
# Voice-message flag of Element (MSC3245) and its audio details (MSC1767 extensible events).
VOICE_FLAG = "org.matrix.msc3245.voice"
AUDIO_DETAILS = "org.matrix.msc1767.audio"
VOICE_FILENAME = "voice-message.ogg"
# Encryption of the rooms the agent creates.
MEGOLM_ALGORITHM = "m.megolm.v1.aes-sha2"
# A new encrypted room is used once the sync shows it encrypted, with the contact in it.
NEW_ROOM_WAIT_SECS = 15.0
NEW_ROOM_POLL_SECS = 0.2
# Messages that could not be decrypted yet: decrypted again after each sync, until their key
# arrives or they are this old. At most this many are kept.
PENDING_DECRYPT_SECS = 24 * 3600.0
MAX_PENDING_DECRYPT = 200
# Devices of a contact seen so far (inbox key-value store), to log the new ones.
DEVICES_KEY = "devices:{}"
# Media placeholders by msgtype (stickers are their own event type).
MEDIA_MSGTYPES = {"m.image": KIND_IMAGE, "m.file": KIND_FILE, "m.video": KIND_VIDEO, "m.location": KIND_LOCATION}
STICKER_EVENT = "m.sticker"
REDACTION_EVENT = "m.room.redaction"
# Text description of a location (MSC3488), better than the body some clients fill with a geo: URI.
LOCATION_DETAILS = "org.matrix.msc3488.location"
# A reply quotes this many characters of the message it answers, at most.
MAX_QUOTE_CHARS = 80
_QUOTED_SENDER = re.compile(r"^\*?\s*<([^>]*)>\s*")


# Called with each new message a contact sent, right after it was stored.
MessageListener = Callable[[InboxMessage, Contact], None]


class MatrixError(RuntimeError):
    """A Matrix request failed."""


def media_key(event: Any) -> str | None:
    """The decryption key of an encrypted attachment event, as stored in the inbox (JSON)."""
    return file_media_key({"key": event.key, "hashes": event.hashes, "iv": event.iv})


def file_media_key(file: dict[str, Any]) -> str | None:
    """The inbox form (JSON) of the decryption key in an attachment's ``file`` object, or None."""
    key = (file.get("key") or {}).get("k")
    sha256 = (file.get("hashes") or {}).get("sha256")
    iv = file.get("iv")
    if not key or not sha256 or not iv:
        return None
    return json.dumps({"k": key, "iv": iv, "sha256": sha256})


def decrypt_media(data: bytes, key: str) -> bytes:
    """Decrypt an attachment downloaded with its stored ``media_key``."""
    try:
        parsed = json.loads(key)
        return decrypt_attachment(data, parsed["k"], parsed["sha256"], parsed["iv"])
    except (EncryptionError, KeyError, TypeError, ValueError) as exc:
        raise MatrixError(f"cannot decrypt the attachment: {exc}") from exc


@dataclass(frozen=True)
class MessageEdit:
    """A contact replaced the text (or the caption) of an earlier message."""

    target: str
    body: str
    filename: str | None = None


@dataclass(frozen=True)
class MessageRedaction:
    """A contact deleted an earlier message."""

    target: str


def _source(event: Any) -> dict[str, Any]:
    source = getattr(event, "source", None)
    return source if isinstance(source, dict) else {}


def _content(event: Any) -> dict[str, Any]:
    content = _source(event).get("content")
    return content if isinstance(content, dict) else {}


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def event_sender(event: Any) -> str | None:
    """The sender of a timeline event (also of events nio could not parse)."""
    return getattr(event, "sender", None) or _source(event).get("sender")


def _relation(content: dict[str, Any]) -> dict[str, Any]:
    relation = content.get("m.relates_to")
    return relation if isinstance(relation, dict) else {}


def _reply_target(content: dict[str, Any]) -> str | None:
    """The event a message replies to, or None (a thread's fallback reply is not a reply)."""
    relation = _relation(content)
    if relation.get("is_falling_back"):
        return None
    target = (relation.get("m.in_reply_to") or {}).get("event_id")
    return target if isinstance(target, str) and target else None


def split_reply_fallback(body: str) -> tuple[str, str | None, str | None]:
    """A reply's body without its quoted fallback: (reply, quoted text, quoted sender).

    The fallback is the leading ``> `` lines (``> <@bob:server> original``) and the blank line after them.
    The HTML fallback (``<mx-reply>`` in ``formatted_body``) is never stored, so it needs no stripping.
    """
    lines = body.split("\n")
    quoted = []
    while lines and lines[0].startswith(">"):
        quoted.append(lines.pop(0)[1:].removeprefix(" "))
    if not quoted:
        return body, None, None
    if lines and not lines[0].strip():
        lines.pop(0)
    sender = _QUOTED_SENDER.match(quoted[0])
    quoted[0] = _QUOTED_SENDER.sub("", quoted[0])
    quote = " ".join(" ".join(quoted).split())
    return "\n".join(lines), quote or None, sender.group(1) if sender else None


def short_quote(text: str) -> str:
    """The start of a message, cut at a word, as a reply quotes it."""
    text = " ".join(text.split())
    if len(text) <= MAX_QUOTE_CHARS:
        return text
    cut = text[:MAX_QUOTE_CHARS].rsplit(" ", 1)[0] or text[:MAX_QUOTE_CHARS]
    return cut.rstrip(",;:") + "…"


def _caption(content: dict[str, Any]) -> tuple[str, str | None]:
    """(caption, file name) of an attachment: with a ``filename``, a different ``body`` is the caption."""
    body = _text(content.get("body")).strip()
    filename = _text(content.get("filename")).strip() or None
    if filename:
        return ("" if body == filename else body), filename
    return "", body or None


def _location_caption(content: dict[str, Any]) -> str:
    """What a shared location says about itself (never its geo: URI)."""
    details = content.get(LOCATION_DETAILS)
    description = _text(details.get("description")).strip() if isinstance(details, dict) else ""
    if description:
        return description
    body = _text(content.get("body")).strip()
    return "" if "geo:" in body else body


def _placeholder(event: Any, room_id: str, sent_at: float) -> InboxMessage | None:
    """The inbox placeholder of a photo, file, video, location or sticker (plain or encrypted)."""
    source = _source(event)
    content = _content(event)
    if source.get("type") == STICKER_EVENT:
        kind = KIND_STICKER
    elif source.get("type") == "m.room.message" and content.get("msgtype") in MEDIA_MSGTYPES:
        kind = MEDIA_MSGTYPES[content["msgtype"]]
    else:
        return None
    event_id = _text(source.get("event_id")) or _text(getattr(event, "event_id", None))
    sender = event_sender(event)
    if not event_id or not sender:
        return None
    if kind == KIND_LOCATION:
        caption, filename = _location_caption(content), None
    elif kind == KIND_STICKER:
        caption, filename = _text(content.get("body")).strip(), None
    else:
        caption, filename = _caption(content)
    file = content.get("file") if isinstance(content.get("file"), dict) else {}
    info = content.get("info") if isinstance(content.get("info"), dict) else {}
    return InboxMessage(
        event_id=event_id,
        room_id=room_id,
        sender=sender,
        kind=kind,
        body=caption,
        sent_at=sent_at,
        # Kept for later; the media itself is never downloaded.
        media_url=_text(content.get("url") or file.get("url")) or None,
        mimetype=_text(info.get("mimetype") or file.get("mimetype")) or None,
        media_key=file_media_key(file) if file else None,
        filename=filename,
    )


def inbox_change(event: Any) -> MessageEdit | MessageRedaction | None:
    """The edit or the redaction of an earlier message that a timeline event is, or None."""
    source = _source(event)
    content = _content(event)
    if source.get("type") == REDACTION_EVENT:
        # The target is at the top level before room version 11, in the content since.
        target = getattr(event, "redacts", None) or source.get("redacts") or content.get("redacts")
        return MessageRedaction(target) if isinstance(target, str) and target else None
    if source.get("type") != "m.room.message":
        return None
    relation = _relation(content)
    target = relation.get("event_id")
    if relation.get("rel_type") != "m.replace" or not isinstance(target, str) or not target:
        return None
    new = content.get("m.new_content")
    if not isinstance(new, dict) or not isinstance(new.get("body"), str):
        return None
    if new.get("msgtype") == "m.location":
        return MessageEdit(target, _location_caption(new))
    if new.get("msgtype") in MEDIA_MSGTYPES:
        caption, filename = _caption(new)
        return MessageEdit(target, caption, filename)
    return MessageEdit(target, new["body"])


def inbox_message(room_id: str, event: Any) -> InboxMessage | None:
    """The inbox form of a timeline event, or None when it is not a message to store.

    Edits and redactions are not messages (``inbox_change``); reactions are ignored.
    """
    source = _source(event)
    sent_at = (getattr(event, "server_timestamp", 0) or source.get("origin_server_ts") or 0) / 1000
    unsigned = source.get("unsigned")
    if (isinstance(unsigned, dict) and unsigned.get("redacted_because")) or inbox_change(event) is not None:
        return None
    if isinstance(event, RoomMessageText | RoomMessageEmote):
        body, reply_to, quote, quoted_sender = event.body, _reply_target(_content(event)), None, None
        if reply_to:
            body, quote, quoted_sender = split_reply_fallback(body)
        return InboxMessage(
            event_id=event.event_id,
            room_id=room_id,
            sender=event.sender,
            kind=KIND_TEXT,
            body=body,
            sent_at=sent_at,
            reply_to=reply_to,
            reply_quote=short_quote(quote) if quote else None,
            # In a direct chat, a quote of somebody else's message quotes the agent's; the gateway
            # settles it from the messages it knows (``MatrixGateway._with_reply_quote``).
            reply_mine=quoted_sender is not None and quoted_sender != event.sender,
        )
    if isinstance(event, RoomMessageAudio):
        info = (event.source.get("content") or {}).get("info") or {}
        duration = info.get("duration")
        return InboxMessage(
            event_id=event.event_id,
            room_id=room_id,
            sender=event.sender,
            kind=KIND_AUDIO,
            body=event.body,
            sent_at=sent_at,
            media_url=event.url,
            mimetype=info.get("mimetype"),
            duration_ms=int(duration) if isinstance(duration, int | float) else None,
        )
    if isinstance(event, RoomEncryptedAudio):
        info = (event.source.get("content") or {}).get("info") or {}
        duration = info.get("duration")
        key = media_key(event)
        if key is None:
            return None
        return InboxMessage(
            event_id=event.event_id,
            room_id=room_id,
            sender=event.sender,
            kind=KIND_AUDIO,
            body=event.body,
            sent_at=sent_at,
            media_url=event.url,
            mimetype=info.get("mimetype") or event.mimetype,
            duration_ms=int(duration) if isinstance(duration, int | float) else None,
            media_key=key,
        )
    return _placeholder(event, room_id, sent_at)


def _inviter(own_user_id: str, invite_state: list[Any]) -> str | None:
    for event in invite_state:
        if isinstance(event, InviteMemberEvent) and event.state_key == own_user_id and event.membership == "invite":
            return event.sender
    return None


class MatrixGateway:
    """The agent's Matrix client: background sync, inbox and sending."""

    def __init__(self, config: MatrixConfig, *, client: AsyncClient | None = None, inbox: Inbox | None = None):
        """Bind the gateway to an account (``client`` and ``inbox`` are injectable for tests)."""
        self.config = config
        self.contacts = ContactBook(config.contacts_file)
        self.inbox = inbox or Inbox(config.db_path)
        # With its own client, the gateway logs its device in before the first sync (encryption).
        self._login_pending = False
        if client is None:
            encryption = config.encryption and ENCRYPTION_ENABLED
            if config.encryption and not ENCRYPTION_ENABLED:
                logger.error("Matrix: encryption needs matrix-nio[e2e] (vodozemac); running unencrypted")
            client = AsyncClient(
                config.homeserver,
                config.user_id,
                device_id=config.device_id or None,
                store_path=str(config.store_dir) if encryption else "",
                config=AsyncClientConfig(
                    encryption_enabled=encryption, store_sync_tokens=False, pickle_key=config.store_key
                ),
            )
            client.access_token = config.access_token
            client.user_id = config.user_id
            self._login_pending = encryption
        self._client = client
        self._task: asyncio.Task | None = None
        # Set once the client knows its rooms (stored rooms are only reused after that).
        self.synced = asyncio.Event()
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._listeners: list[MessageListener] = []
        # The first sync of the process loads every room's state (encrypted sends need it).
        self._full_state_loaded = False
        # Undecryptable events by event id: (room id, event, received at).
        self._pending_decrypt: OrderedDict[str, tuple[str, MegolmEvent, float]] = OrderedDict()
        self._encrypted_rooms_warned: set[str] = set()
        self._store_lock: IO[str] | None = None

    @property
    def encrypted(self) -> bool:
        """Whether end-to-end encryption runs (the device keys are loaded)."""
        return getattr(self._client, "olm", None) is not None

    def lock_store(self) -> None:
        """Take the key store for this process; MatrixError when another process has it.

        Two processes sharing the device keys would break each other's sessions.
        """
        if not self._login_pending:
            return
        self.config.store_dir.mkdir(parents=True, exist_ok=True)
        lock = open(self.config.store_dir / "store.lock", "w")  # noqa: SIM115 - held until close()
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock.close()
            raise MatrixError(
                f"another process uses the encryption keys in {self.config.store_dir} (run one server worker)"
            ) from exc
        self._store_lock = lock

    async def _login(self) -> None:
        """Load the device keys of the access token's device (encryption)."""
        response = await self._client.whoami()
        if not isinstance(response, WhoamiResponse):
            raise MatrixError(f"whoami: {getattr(response, 'message', response)}")
        device_id = response.device_id or ""
        if not device_id:
            raise MatrixError("the access token has no device; log the agent in again to get a new token")
        if self.config.device_id and self.config.device_id != device_id:
            logger.warning(
                f"Matrix: MATRIX_DEVICE_ID is {self.config.device_id}, but the token belongs to {device_id}; "
                f"using {device_id}"
            )
        self._login_pending = False
        store_file = self.config.store_dir / f"{self.config.user_id}_{device_id}.db"
        if not store_file.exists() and await self._device_has_keys(device_id):
            # Another client holds this device's keys (an Element session's token, or a lost store):
            # keys of our own would clash with them, so the agent stays unencrypted.
            logger.error(
                f"Matrix: device {device_id} of the access token already has encryption keys (the token of an "
                "Element session, or lost keys); running unencrypted. Get a token of the agent's own with the "
                "login API to read encrypted rooms."
            )
            return
        self._client.restore_login(self.config.user_id, device_id, self.config.access_token)
        logger.info(f"Matrix: end-to-end encryption on (device {device_id})")

    async def _device_has_keys(self, device_id: str) -> bool:
        """Whether the homeserver already has encryption keys for the agent's ``device_id``."""
        method, path, data = Api.keys_query(self.config.access_token, {self.config.user_id})
        response = await self._client.send(method, path, data)
        if response.status != 200:
            raise MatrixError(f"keys query: HTTP {response.status}")
        body = await response.json()
        return device_id in ((body.get("device_keys") or {}).get(self.config.user_id) or {})

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start syncing in the background."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._sync_loop(), name="matrix-sync")
            logger.info(f"Matrix: syncing as {self.config.user_id} on {self.config.homeserver}")

    async def close(self) -> None:
        """Stop syncing and close the connection."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._client.close()
        self.inbox.close()
        if self._store_lock is not None:
            self._store_lock.close()
            self._store_lock = None

    def add_listener(self, listener: MessageListener) -> Callable[[], None]:
        """Call ``listener`` with each new message from a contact; returns the function that removes it.

        Listeners run inside the sync loop: they must return at once (schedule work, do not await it).
        """
        self._listeners.append(listener)

        def remove() -> None:
            with contextlib.suppress(ValueError):
                self._listeners.remove(listener)

        return remove

    def _notify(self, message: InboxMessage, contact: Contact) -> None:
        for listener in list(self._listeners):
            try:
                listener(message, contact)
            except Exception as exc:
                logger.opt(exception=exc).warning("Matrix: a new-message listener failed")

    # ---------------------------------------------------------------- sync
    async def _sync_loop(self) -> None:
        delay = RETRY_MIN_SECS
        while True:
            try:
                await self.sync_once()
                delay = RETRY_MIN_SECS
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.opt(exception=exc).warning(f"Matrix: sync failed; retrying in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, RETRY_MAX_SECS)

    async def sync_once(self) -> None:
        """One sync round: join contacts' invitations and store their new messages."""
        if self._login_pending:
            await self._login()
        since = await asyncio.to_thread(self.inbox.get, SYNC_TOKEN_KEY)
        # The first sync only catches up: history from before the agent had an inbox is not imported.
        # The first one of the process also loads the state of every room (members, encryption).
        response = await self._client.sync(
            timeout=SYNC_TIMEOUT_MS if since and self._full_state_loaded else 0,
            since=since,
            full_state=since is None or not self._full_state_loaded,
        )
        if not isinstance(response, SyncResponse):
            raise MatrixError(f"sync: {getattr(response, 'message', response)}")
        self._full_state_loaded = True
        if self.encrypted:
            await self._manage_keys()
        for room_id, info in response.rooms.invite.items():
            await self._on_invite(room_id, info.invite_state)
        if since is not None:
            for room_id, info in response.rooms.join.items():
                for event in info.timeline.events:
                    await self._on_event(room_id, event)
            await self._retry_decrypt()
        await asyncio.to_thread(self.inbox.put, SYNC_TOKEN_KEY, response.next_batch)
        self.synced.set()

    # ---------------------------------------------------------- encryption
    async def _manage_keys(self) -> None:
        """Share and fetch keys after a sync, as nio's own sync loop does."""
        client = self._client
        try:
            await client.send_to_device_messages()
            if client.should_upload_keys:
                response = await client.keys_upload()
                if isinstance(response, KeysUploadError):
                    logger.error(
                        f"Matrix: the homeserver refused the device keys ({response.message}). The token "
                        "must be the agent's own (not an Element session's), with the keys in "
                        f"{self.config.store_dir}; when they were lost, log the agent in again for a new device."
                    )
            if client.should_query_keys:
                await client.keys_query()
                await self._check_new_devices()
            if client.should_claim_keys:
                await client.keys_claim(client.get_users_for_key_claiming())
        except Exception as exc:
            logger.opt(exception=exc).warning("Matrix: key management failed; retrying after the next sync")

    async def _check_new_devices(self) -> None:
        """Log the devices contacts added since they were last seen (all of them get the keys)."""
        for contact in self.contacts.all():
            current = {device.id for device in self._client.device_store.active_user_devices(contact.matrix_id)}
            if not current:
                continue
            key = DEVICES_KEY.format(contact.matrix_id)
            stored = await asyncio.to_thread(self.inbox.get, key)
            known = set(json.loads(stored)) if stored else None
            if known is not None and current <= known:
                continue
            if known is not None:
                logger.warning(
                    f"Matrix: {contact.name} has a new device ({', '.join(sorted(current - known))}); "
                    "messages are encrypted for it too"
                )
            await asyncio.to_thread(self.inbox.put, key, json.dumps(sorted(current | (known or set()))))

    async def _undecryptable(self, room_id: str, event: MegolmEvent) -> None:
        """Ask the sender's devices for the key, and try again after the next syncs."""
        if not self.encrypted:
            if room_id not in self._encrypted_rooms_warned:
                self._encrypted_rooms_warned.add(room_id)
                logger.warning(f"Matrix: ignoring encrypted messages in {room_id} (MATRIX_ENCRYPTION is off)")
            return
        logger.warning(f"Matrix: cannot decrypt {event.event_id} from {event.sender} yet; asking for its key")
        self._pending_decrypt[event.event_id] = (room_id, event, time.monotonic())
        while len(self._pending_decrypt) > MAX_PENDING_DECRYPT:
            self._pending_decrypt.popitem(last=False)
        with contextlib.suppress(LocalProtocolError):
            await self._client.request_room_key(event)

    async def _retry_decrypt(self) -> None:
        now = time.monotonic()
        for event_id, (room_id, event, received) in list(self._pending_decrypt.items()):
            if now - received > PENDING_DECRYPT_SECS:
                del self._pending_decrypt[event_id]
                continue
            try:
                decrypted = self._client.decrypt_event(event)
            except EncryptionError:
                continue
            del self._pending_decrypt[event_id]
            await self._on_event(room_id, decrypted)

    async def _on_invite(self, room_id: str, invite_state: list[Any]) -> None:
        inviter = _inviter(self.config.user_id, invite_state)
        contact = self.contacts.by_matrix_id(inviter) if inviter else None
        if contact is None:
            logger.info(f"Matrix: ignoring an invitation to {room_id} from {inviter or 'unknown'} (not a contact)")
            return
        response = await self._client.join(room_id)
        if isinstance(response, JoinError):
            logger.warning(f"Matrix: could not join {room_id} on {contact.name}'s invitation: {response.message}")
            return
        await asyncio.to_thread(self.inbox.set_room, contact.matrix_id, room_id)
        logger.info(f"Matrix: joined {room_id} on {contact.name}'s invitation")

    def _is_direct(self, room_id: str) -> bool:
        room = self._client.rooms.get(room_id)
        # Unknown room state: the room was just joined, trust it.
        return room is None or room.member_count <= MAX_DM_MEMBERS

    async def _on_event(self, room_id: str, event: Any) -> None:
        sender = event_sender(event)
        if sender == self.config.user_id:
            return
        if isinstance(event, MegolmEvent):
            if self.contacts.by_matrix_id(event.sender) is not None and self._is_direct(room_id):
                await self._undecryptable(room_id, event)
            return
        contact = self.contacts.by_matrix_id(sender) if sender else None
        if contact is None or not self._is_direct(room_id):
            return
        change = inbox_change(event)
        if change is not None:
            # Not a new message: listeners are not told about it.
            await self._apply_change(room_id, sender, contact, change)
            return
        message = inbox_message(room_id, event)
        if message is None:
            return
        if message.reply_to:
            message = await self._with_reply_quote(message)
        if await asyncio.to_thread(self.inbox.add, message):
            # Replies go where the contact last wrote.
            await asyncio.to_thread(self.inbox.set_room, contact.matrix_id, room_id)
            logger.info(f"Matrix: new {message.kind} message from {contact.name}")
            self._notify(message, contact)

    async def _apply_change(
        self, room_id: str, sender: str, contact: Contact, change: MessageEdit | MessageRedaction
    ) -> None:
        """Edit or delete a stored message; a change to a message that is not stored is ignored."""
        if isinstance(change, MessageEdit):
            edit = (change.target, room_id, sender, change.body, change.filename)
            done = await asyncio.to_thread(self.inbox.edit, *edit)
            what = "edited"
        else:
            done = await asyncio.to_thread(self.inbox.remove, change.target, room_id, sender)
            what = "deleted"
        if done:
            logger.info(f"Matrix: {contact.name} {what} a message")
        else:
            logger.debug(f"Matrix: {contact.name} {what} {change.target}, which is not in the inbox; ignored")

    async def _with_reply_quote(self, message: InboxMessage) -> InboxMessage:
        """A reply with the quote of the message it answers: the agent's own, a stored one, or its fallback."""
        sent = await asyncio.to_thread(self.inbox.sent_body, message.reply_to)
        if sent is not None:
            return replace(message, reply_quote=short_quote(sent) or None, reply_mine=True)
        original = await asyncio.to_thread(self.inbox.message, message.reply_to)
        if original is not None and original.body:
            return replace(message, reply_quote=short_quote(original.body), reply_mine=False)
        return message

    async def _remember_sent(self, room_id: str, event_id: str, body: str) -> None:
        """Keep what the agent sent, to quote it when a contact replies to it."""
        try:
            await asyncio.to_thread(self.inbox.add_sent, event_id, room_id, body)
        except Exception as exc:
            logger.opt(exception=exc).debug("Matrix: could not keep a sent message")

    # ------------------------------------------------------------- sending
    async def _request(self, what: str, call) -> Any:
        """Run a nio request, waiting out rate limits."""
        for _ in range(MAX_RATE_LIMIT_RETRIES):
            response = await call()
            retry_ms = getattr(response, "retry_after_ms", None)
            if isinstance(response, ErrorResponse) and response.status_code == "M_LIMIT_EXCEEDED" and retry_ms:
                await asyncio.sleep(retry_ms / 1000)
                continue
            if isinstance(response, ErrorResponse):
                raise MatrixError(f"{what}: {response.message}")
            return response
        raise MatrixError(f"{what}: rate limited")

    async def _direct_room(self, contact: Contact) -> str:
        """The room used with ``contact``, created (unencrypted) the first time."""
        lock = self._room_locks.setdefault(contact.matrix_id, asyncio.Lock())
        async with lock:
            room_id = await asyncio.to_thread(self.inbox.room_for, contact.matrix_id)
            room = self._client.rooms.get(room_id) if room_id else None
            # Reused while the contact is still in it (joined, or not yet answered the invitation).
            if room is not None and contact.matrix_id in {*room.users, *room.invited_users}:
                return room_id
            encrypted = self.encrypted
            initial_state = (
                [{"type": "m.room.encryption", "state_key": "", "content": {"algorithm": MEGOLM_ALGORITHM}}]
                if encrypted
                else []
            )
            response = await self._request(
                f"create a room with {contact.matrix_id}",
                lambda: self._client.room_create(
                    is_direct=True,
                    invite=[contact.matrix_id],
                    preset=RoomPreset.trusted_private_chat,
                    initial_state=initial_state,
                ),
            )
            await asyncio.to_thread(self.inbox.set_room, contact.matrix_id, response.room_id)
            kind = "an encrypted" if encrypted else "a"
            logger.info(f"Matrix: created {kind} room {response.room_id} with {contact.name}")
            if encrypted:
                await self._wait_for_room(response.room_id, contact)
            return response.room_id

    async def _wait_for_room(self, room_id: str, contact: Contact) -> None:
        """Wait until the sync shows a new room encrypted, with the contact in it (who gets the key)."""
        deadline = time.monotonic() + NEW_ROOM_WAIT_SECS
        while time.monotonic() < deadline:
            room = self._client.rooms.get(room_id)
            if room is not None and room.encrypted and contact.matrix_id in {*room.users, *room.invited_users}:
                return
            await asyncio.sleep(NEW_ROOM_POLL_SECS)
        raise MatrixError(f"the new room with {contact.matrix_id} did not sync in time")

    async def _wait_synced(self) -> None:
        try:
            await asyncio.wait_for(self.synced.wait(), FIRST_SYNC_WAIT_SECS)
        except TimeoutError as exc:
            raise MatrixError("not connected to the homeserver yet") from exc

    async def _send_message(self, contact: Contact, room_id: str, content: dict[str, Any]) -> str:
        response = await self._request(
            f"send to {contact.matrix_id}",
            # Encrypted for every device of the contact, verified or not (trust on first use).
            lambda: self._client.room_send(room_id, "m.room.message", content, ignore_unverified_devices=True),
        )
        return response.event_id

    def _room_encrypted(self, room_id: str) -> bool:
        room = self._client.rooms.get(room_id)
        return self.encrypted and room is not None and room.encrypted

    async def send_text(self, contact: Contact, text: str) -> str:
        """Send ``text`` to ``contact``; returns the event id."""
        async with _send_errors(contact):
            await self._wait_synced()
            room_id = await self._direct_room(contact)
            event_id = await self._send_message(contact, room_id, {"msgtype": "m.text", "body": text})
        await self._remember_sent(room_id, event_id, text)
        return event_id

    async def send_voice(
        self, contact: Contact, audio: bytes, *, duration_ms: int, waveform: list[int], body: str
    ) -> str:
        """Upload an Ogg/Opus recording and send it to ``contact`` as a voice message; returns the event id.

        ``body`` is the text that clients without voice messages show (and Element shows as a caption).
        """
        async with _send_errors(contact):
            await self._wait_synced()
            room_id = await self._direct_room(contact)
            # An encrypted room gets an encrypted attachment: the homeserver only stores ciphertext.
            encrypt = self._room_encrypted(room_id)
            response, keys = await self._client.upload(
                lambda *_: io.BytesIO(audio),
                content_type=OGG_MIMETYPE,
                filename=VOICE_FILENAME,
                filesize=len(audio),
                encrypt=encrypt,
            )
            if not isinstance(response, UploadResponse):
                raise MatrixError(f"upload a voice message: {getattr(response, 'message', response)}")
            content: dict[str, Any] = {
                "msgtype": "m.audio",
                "body": body,
                "filename": VOICE_FILENAME,
                "info": {"mimetype": OGG_MIMETYPE, "size": len(audio), "duration": duration_ms},
                AUDIO_DETAILS: {"duration": duration_ms, "waveform": waveform},
                VOICE_FLAG: {},
            }
            if encrypt:
                content["file"] = {"url": response.content_uri, "mimetype": OGG_MIMETYPE, **(keys or {})}
            else:
                content["url"] = response.content_uri
            event_id = await self._send_message(contact, room_id, content)
        await self._remember_sent(room_id, event_id, body)
        return event_id

    async def download_media(self, message: InboxMessage) -> bytes:
        """The media of a received message, decrypted when it was sent encrypted."""
        if not message.media_url:
            raise MatrixError("the message has no media")
        data = await self.download(message.media_url)
        return decrypt_media(data, message.media_key) if message.media_key else data

    async def download(self, mxc_url: str) -> bytes:
        """Download received media (authenticated media API, required by matrix.org)."""
        response = await self._client.download(mxc=mxc_url)
        if isinstance(response, DownloadError) or not hasattr(response, "body"):
            raise MatrixError(f"download {mxc_url}: {getattr(response, 'message', response)}")
        if len(response.body) > MAX_MEDIA_BYTES:
            raise MatrixError(f"download {mxc_url}: larger than {MAX_MEDIA_BYTES} bytes")
        return response.body

    # ---------------------------------------------------------------- inbox
    async def mark_read(self, messages: list[InboxMessage]) -> None:
        """Mark messages read, here and for their senders (read receipts)."""
        await asyncio.to_thread(self.inbox.mark_read, [m.event_id for m in messages])
        latest: dict[str, InboxMessage] = {}
        for message in messages:
            if message.room_id not in latest or message.sent_at >= latest[message.room_id].sent_at:
                latest[message.room_id] = message
        for room_id, message in latest.items():
            try:
                await self._client.room_read_markers(room_id, message.event_id, message.event_id)
            except Exception as exc:
                logger.opt(exception=exc).debug(f"Matrix: could not send a read receipt in {room_id}")


@contextlib.asynccontextmanager
async def _send_errors(contact: Contact):
    """Any failure of a send is a MatrixError: what nio raises (encryption, connection) is not sent."""
    try:
        yield
    except MatrixError:
        raise
    except Exception as exc:
        raise MatrixError(f"send to {contact.matrix_id}: {exc}") from exc


_gateway: MatrixGateway | None = None


def get_gateway() -> MatrixGateway | None:
    """The running gateway, or None when messaging is off."""
    return _gateway


async def start_gateway(config: MatrixConfig) -> MatrixGateway | None:
    """Start the process-wide gateway when ``config`` is enabled."""
    global _gateway
    if not config.enabled or _gateway is not None:
        return _gateway
    try:
        gateway = MatrixGateway(config)
    except Exception as exc:
        logger.opt(exception=exc).error("Matrix messaging unavailable")
        return None
    try:
        gateway.lock_store()
    except MatrixError as exc:
        logger.error(f"Matrix messaging unavailable in this process: {exc}")
        await gateway.close()
        return None
    _gateway = gateway
    _gateway.start()
    return _gateway


async def stop_gateway() -> None:
    """Stop the process-wide gateway."""
    global _gateway
    gateway, _gateway = _gateway, None
    if gateway is not None:
        await gateway.close()

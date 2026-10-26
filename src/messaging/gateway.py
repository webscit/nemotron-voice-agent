# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent's Matrix account: one client per server process, shared by all sessions.

The gateway syncs in the background, stores what contacts send in the inbox, and
sends messages to contacts in a direct room it creates (unencrypted) or that the
contact invited it to. Messages from anybody who is not in the contacts file are
ignored, and invitations from them are left pending.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from typing import Any

from loguru import logger
from nio import (
    AsyncClient,
    AsyncClientConfig,
    DownloadError,
    ErrorResponse,
    InviteMemberEvent,
    JoinError,
    RoomMessageAudio,
    RoomMessageEmote,
    RoomMessageText,
    RoomPreset,
    SyncResponse,
    UploadResponse,
)

from messaging.config import MatrixConfig
from messaging.contacts import Contact, ContactBook
from messaging.inbox import KIND_AUDIO, KIND_TEXT, SYNC_TOKEN_KEY, Inbox, InboxMessage
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


class MatrixError(RuntimeError):
    """A Matrix request failed."""


def inbox_message(room_id: str, event: Any) -> InboxMessage | None:
    """The inbox form of a timeline event, or None when it is not a text or audio message."""
    sent_at = (getattr(event, "server_timestamp", 0) or 0) / 1000
    if isinstance(event, RoomMessageText | RoomMessageEmote):
        return InboxMessage(
            event_id=event.event_id,
            room_id=room_id,
            sender=event.sender,
            kind=KIND_TEXT,
            body=event.body,
            sent_at=sent_at,
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
    return None


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
        if client is None:
            client = AsyncClient(
                config.homeserver,
                config.user_id,
                device_id=config.device_id or None,
                config=AsyncClientConfig(encryption_enabled=False, store_sync_tokens=False),
            )
            client.access_token = config.access_token
            client.user_id = config.user_id
        self._client = client
        self._task: asyncio.Task | None = None
        # Set once the client knows its rooms (stored rooms are only reused after that).
        self.synced = asyncio.Event()
        self._room_locks: dict[str, asyncio.Lock] = {}

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
        since = await asyncio.to_thread(self.inbox.get, SYNC_TOKEN_KEY)
        # The first sync only catches up: history from before the agent had an inbox is not imported.
        response = await self._client.sync(
            timeout=SYNC_TIMEOUT_MS if since else 0, since=since, full_state=since is None
        )
        if not isinstance(response, SyncResponse):
            raise MatrixError(f"sync: {getattr(response, 'message', response)}")
        for room_id, info in response.rooms.invite.items():
            await self._on_invite(room_id, info.invite_state)
        if since is not None:
            for room_id, info in response.rooms.join.items():
                for event in info.timeline.events:
                    await self._on_event(room_id, event)
        await asyncio.to_thread(self.inbox.put, SYNC_TOKEN_KEY, response.next_batch)
        self.synced.set()

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
        if getattr(event, "sender", None) == self.config.user_id:
            return
        message = inbox_message(room_id, event)
        if message is None:
            return
        contact = self.contacts.by_matrix_id(message.sender)
        if contact is None or not self._is_direct(room_id):
            return
        if await asyncio.to_thread(self.inbox.add, message):
            # Replies go where the contact last wrote.
            await asyncio.to_thread(self.inbox.set_room, contact.matrix_id, room_id)
            logger.info(f"Matrix: new {message.kind} message from {contact.name}")

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
            response = await self._request(
                f"create a room with {contact.matrix_id}",
                lambda: self._client.room_create(
                    is_direct=True, invite=[contact.matrix_id], preset=RoomPreset.trusted_private_chat
                ),
            )
            await asyncio.to_thread(self.inbox.set_room, contact.matrix_id, response.room_id)
            logger.info(f"Matrix: created room {response.room_id} with {contact.name}")
            return response.room_id

    async def _wait_synced(self) -> None:
        try:
            await asyncio.wait_for(self.synced.wait(), FIRST_SYNC_WAIT_SECS)
        except TimeoutError as exc:
            raise MatrixError("not connected to the homeserver yet") from exc

    async def _send_message(self, contact: Contact, content: dict[str, Any]) -> str:
        room_id = await self._direct_room(contact)
        response = await self._request(
            f"send to {contact.matrix_id}",
            lambda: self._client.room_send(room_id, "m.room.message", content),
        )
        return response.event_id

    async def send_text(self, contact: Contact, text: str) -> str:
        """Send ``text`` to ``contact``; returns the event id."""
        await self._wait_synced()
        return await self._send_message(contact, {"msgtype": "m.text", "body": text})

    async def send_voice(
        self, contact: Contact, audio: bytes, *, duration_ms: int, waveform: list[int], body: str
    ) -> str:
        """Upload an Ogg/Opus recording and send it to ``contact`` as a voice message; returns the event id.

        ``body`` is the text that clients without voice messages show (and Element shows as a caption).
        """
        await self._wait_synced()
        response, _ = await self._client.upload(
            lambda *_: io.BytesIO(audio), content_type=OGG_MIMETYPE, filename=VOICE_FILENAME, filesize=len(audio)
        )
        if not isinstance(response, UploadResponse):
            raise MatrixError(f"upload a voice message: {getattr(response, 'message', response)}")
        return await self._send_message(
            contact,
            {
                "msgtype": "m.audio",
                "body": body,
                "filename": VOICE_FILENAME,
                "url": response.content_uri,
                "info": {"mimetype": OGG_MIMETYPE, "size": len(audio), "duration": duration_ms},
                AUDIO_DETAILS: {"duration": duration_ms, "waveform": waveform},
                VOICE_FLAG: {},
            },
        )

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
        _gateway = MatrixGateway(config)
    except Exception as exc:
        logger.opt(exception=exc).error("Matrix messaging unavailable")
        return None
    _gateway.start()
    return _gateway


async def stop_gateway() -> None:
    """Stop the process-wide gateway."""
    global _gateway
    gateway, _gateway = _gateway, None
    if gateway is not None:
        await gateway.close()

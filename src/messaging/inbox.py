# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite inbox: received messages, the DM room of each contact and the sync token.

Besides text and voice messages, the inbox keeps a placeholder for the media a
contact sends (photo, file, video, location, sticker): its caption and file name,
never the media itself. Edits replace the stored text, redactions delete the
message, and a reply remembers a short quote of the message it answers. The
agent's own recent messages are kept too (``sent``), to quote them when a
contact replies to one.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rooms (
    matrix_id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    event_id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    sender TEXT NOT NULL,
    kind TEXT NOT NULL,
    body TEXT NOT NULL,
    media_url TEXT,
    mimetype TEXT,
    duration_ms INTEGER,
    media_key TEXT,
    transcript TEXT,
    filename TEXT,
    reply_to TEXT,
    reply_quote TEXT,
    reply_mine INTEGER,
    sent_at REAL NOT NULL,
    received_at REAL NOT NULL,
    read_at REAL
);
CREATE INDEX IF NOT EXISTS messages_sender ON messages (sender, sent_at);
CREATE TABLE IF NOT EXISTS sent (
    event_id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    body TEXT NOT NULL,
    sent_at REAL NOT NULL
);
"""
# Columns added after the first release: (name, SQL type), added to older databases.
_ADDED_COLUMNS = (
    ("media_key", "TEXT"),
    ("transcript", "TEXT"),
    ("filename", "TEXT"),
    ("reply_to", "TEXT"),
    ("reply_quote", "TEXT"),
    ("reply_mine", "INTEGER"),
)
# The agent's own messages kept to quote them in replies (the latest ones).
MAX_SENT = 500

SYNC_TOKEN_KEY = "sync_token"
KIND_TEXT = "text"
KIND_AUDIO = "audio"
# Media that is only announced, never downloaded: ``body`` is its caption ("" without one).
KIND_IMAGE = "image"
KIND_FILE = "file"
KIND_VIDEO = "video"
KIND_LOCATION = "location"
KIND_STICKER = "sticker"
MEDIA_KINDS = frozenset({KIND_IMAGE, KIND_FILE, KIND_VIDEO, KIND_LOCATION, KIND_STICKER})


@dataclass(frozen=True)
class InboxMessage:
    """One received message."""

    event_id: str
    room_id: str
    sender: str
    kind: str
    body: str
    sent_at: float
    media_url: str | None = None
    mimetype: str | None = None
    duration_ms: int | None = None
    read_at: float | None = None
    # Decryption key of an encrypted attachment (JSON: k, iv, sha256).
    media_key: str | None = None
    # What the ASR heard in a voice message ("" when it heard nothing), once transcribed.
    transcript: str | None = None
    # File name of an attachment (a file, image or video), when the sender gave one.
    filename: str | None = None
    # A reply: the event it answers, a short quote of that message (None when unknown),
    # and whether that message was the agent's own.
    reply_to: str | None = None
    reply_quote: str | None = None
    reply_mine: bool = False


class Inbox:
    """Thread-safe access to the inbox database."""

    def __init__(self, path: Path):
        """Open (and create) the database at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(messages)")}
            for name, kind in _ADDED_COLUMNS:
                if name not in columns:
                    self._conn.execute(f"ALTER TABLE messages ADD COLUMN {name} {kind}")

    def close(self) -> None:
        """Close the database."""
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ kv
    def get(self, key: str) -> str | None:
        """A stored value, or None."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def put(self, key: str, value: str) -> None:
        """Store a value."""
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))

    # --------------------------------------------------------------- rooms
    def room_for(self, matrix_id: str) -> str | None:
        """The DM room used with a contact, or None."""
        with self._lock:
            row = self._conn.execute("SELECT room_id FROM rooms WHERE matrix_id = ?", (matrix_id,)).fetchone()
        return row["room_id"] if row else None

    def set_room(self, matrix_id: str, room_id: str) -> None:
        """Use ``room_id`` for messages with a contact from now on."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO rooms (matrix_id, room_id, updated_at) VALUES (?, ?, ?)",
                (matrix_id, room_id, time.time()),
            )

    # ------------------------------------------------------------ messages
    def add(self, message: InboxMessage) -> bool:
        """Store a received message; False when it was already stored."""
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO messages (event_id, room_id, sender, kind, body, media_url, mimetype, "
                "duration_ms, media_key, filename, reply_to, reply_quote, reply_mine, sent_at, received_at, read_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message.event_id,
                    message.room_id,
                    message.sender,
                    message.kind,
                    message.body,
                    message.media_url,
                    message.mimetype,
                    message.duration_ms,
                    message.media_key,
                    message.filename,
                    message.reply_to,
                    message.reply_quote,
                    int(message.reply_mine),
                    message.sent_at,
                    time.time(),
                    message.read_at,
                ),
            )
        return cursor.rowcount > 0

    def messages(self, senders: list[str], *, unread_only: bool = True, limit: int = 10) -> list[InboxMessage]:
        """Messages from ``senders``, oldest first: the oldest unread ones, or else the latest ones."""
        if not senders:
            return []
        placeholders = ",".join("?" * len(senders))
        if unread_only:
            sql = (
                f"SELECT * FROM messages WHERE sender IN ({placeholders}) AND read_at IS NULL "
                "ORDER BY sent_at ASC LIMIT ?"
            )
        else:
            sql = (
                f"SELECT * FROM (SELECT * FROM messages WHERE sender IN ({placeholders}) "
                "ORDER BY sent_at DESC LIMIT ?) ORDER BY sent_at ASC"
            )
        with self._lock:
            rows = self._conn.execute(sql, (*senders, limit)).fetchall()
        return [_message(row) for row in rows]

    def unread_counts(self, senders: list[str]) -> dict[str, int]:
        """Number of unread messages per sender (senders without any are left out)."""
        if not senders:
            return {}
        placeholders = ",".join("?" * len(senders))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT sender, COUNT(*) AS n FROM messages WHERE sender IN ({placeholders}) AND read_at IS NULL "
                "GROUP BY sender",
                tuple(senders),
            ).fetchall()
        return {row["sender"]: row["n"] for row in rows}

    def unread_among(self, event_ids: list[str]) -> set[str]:
        """The messages of ``event_ids`` that are stored and still unread."""
        if not event_ids:
            return set()
        placeholders = ",".join("?" * len(event_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT event_id FROM messages WHERE read_at IS NULL AND event_id IN ({placeholders})",
                tuple(event_ids),
            ).fetchall()
        return {row["event_id"] for row in rows}

    def message(self, event_id: str) -> InboxMessage | None:
        """The stored message ``event_id``, or None."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM messages WHERE event_id = ?", (event_id,)).fetchone()
        return _message(row) if row else None

    def edit(self, event_id: str, room_id: str, sender: str, body: str, filename: str | None = None) -> bool:
        """Replace the text (or caption) of a stored message; False when there is no such message.

        Only the message's sender may edit it, in the room it was sent to. Its read state is kept.
        """
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE messages SET body = ?, filename = COALESCE(?, filename) "
                "WHERE event_id = ? AND room_id = ? AND sender = ?",
                (body, filename, event_id, room_id, sender),
            )
        return cursor.rowcount > 0

    def remove(self, event_id: str, room_id: str, sender: str) -> bool:
        """Delete a stored message its sender redacted; False when there is no such message."""
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM messages WHERE event_id = ? AND room_id = ? AND sender = ?", (event_id, room_id, sender)
            )
        return cursor.rowcount > 0

    def add_sent(self, event_id: str, room_id: str, body: str) -> None:
        """Remember a message the agent sent (the latest ``MAX_SENT`` are kept)."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO sent (event_id, room_id, body, sent_at) VALUES (?, ?, ?, ?)",
                (event_id, room_id, body, time.time()),
            )
            self._conn.execute(
                "DELETE FROM sent WHERE event_id NOT IN (SELECT event_id FROM sent ORDER BY sent_at DESC LIMIT ?)",
                (MAX_SENT,),
            )

    def sent_body(self, event_id: str) -> str | None:
        """The text of a message the agent sent, or None when it is not kept."""
        with self._lock:
            row = self._conn.execute("SELECT body FROM sent WHERE event_id = ?", (event_id,)).fetchone()
        return row["body"] if row else None

    def set_transcript(self, event_id: str, transcript: str) -> None:
        """Store what the ASR heard in a voice message."""
        with self._lock, self._conn:
            self._conn.execute("UPDATE messages SET transcript = ? WHERE event_id = ?", (transcript, event_id))

    def mark_read(self, event_ids: list[str]) -> None:
        """Mark messages as read."""
        if not event_ids:
            return
        placeholders = ",".join("?" * len(event_ids))
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE messages SET read_at = ? WHERE read_at IS NULL AND event_id IN ({placeholders})",
                (time.time(), *event_ids),
            )


def _message(row: sqlite3.Row) -> InboxMessage:
    return InboxMessage(
        event_id=row["event_id"],
        room_id=row["room_id"],
        sender=row["sender"],
        kind=row["kind"],
        body=row["body"],
        sent_at=row["sent_at"],
        media_url=row["media_url"],
        mimetype=row["mimetype"],
        duration_ms=row["duration_ms"],
        read_at=row["read_at"],
        media_key=row["media_key"],
        transcript=row["transcript"],
        filename=row["filename"],
        reply_to=row["reply_to"],
        reply_quote=row["reply_quote"],
        reply_mine=bool(row["reply_mine"]),
    )

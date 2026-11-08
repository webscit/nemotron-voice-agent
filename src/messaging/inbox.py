# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQLite inbox: received messages, the DM room of each contact and the sync token."""

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
    sent_at REAL NOT NULL,
    received_at REAL NOT NULL,
    read_at REAL
);
CREATE INDEX IF NOT EXISTS messages_sender ON messages (sender, sent_at);
"""
# Columns added after the first release: (name, SQL type), added to older databases.
_ADDED_COLUMNS = (("media_key", "TEXT"), ("transcript", "TEXT"))

SYNC_TOKEN_KEY = "sync_token"
KIND_TEXT = "text"
KIND_AUDIO = "audio"


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
                "duration_ms, media_key, sent_at, received_at, read_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
    )

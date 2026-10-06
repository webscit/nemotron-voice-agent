# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live side of voice and face ID: who is speaking in each user turn, and who is in view.

The client computes speaker embeddings and streams ``speaker-update`` messages
(see ``docs/voice-id-protocol.md``). This module keeps the latest estimate,
attributes each user turn to a person when the turn is committed, swaps the
per-person memory block of the pinned prompt, and enrolls unknown speakers
through the server-side ``enroll_speaker`` tool.

A client with a camera also confirms the voice with the speaker's face (tier
``verified``) and reports who is in view with ``presence-update`` messages. The
people in view are named to the LLM, and a recognised person who comes into
view while nobody talks is greeted once per session. Presence never loads
memories and never changes who the speaker is.

- ``VoiceIdTracker`` is the pure session state (no Pipecat, no database).
- ``VoiceIdSession`` wires the tracker to the LLM context, the people store and
  the RTVI channel.
- ``SpeakerTurnProcessor`` commits the turn right before the context reaches
  the LLM.

Everything is best effort: a missing store, a malformed update or a database
error never blocks a turn.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMContextFrame,
    LLMRunFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from examples.shared.person_memory import (
    PersonContext,
    load_person_context,
    record_memory_uses,
    record_turn_speaker,
)

if TYPE_CHECKING:
    import numpy as np
    from pipecat.services.llm_service import FunctionCallParams

SPEAKER_UPDATE_MESSAGE = "speaker-update"
PRESENCE_UPDATE_MESSAGE = "presence-update"
SPEAKER_ENROLLED_MESSAGE = "speaker-enrolled"
VOICE_ID_ADDON_KEY = "voice_id_addon"
FACE_ID_ADDON_KEY = "face_id_addon"
ENROLL_TOOL_NAME = "enroll_speaker"

# Voice and the face linked to the speaker agree on the person.
TIER_VERIFIED = "verified"
TIER_HIGH = "high"
TIER_LOW = "low"
TIER_UNKNOWN = "unknown"
TIER_NONE = "none"
TIERS = (TIER_VERIFIED, TIER_HIGH, TIER_LOW, TIER_UNKNOWN, TIER_NONE)
# Tiers that identify a person well enough for their memories and policies.
TRUSTED_TIERS = (TIER_VERIFIED, TIER_HIGH)
FACE_TIERS = (TIER_HIGH, TIER_LOW, TIER_UNKNOWN)
FACE_LINKS = ("doa", "single")

# The client's VAD and data channel are faster than the server's turn start, so
# an update that arrives slightly before the turn opens still belongs to it.
FRESH_GRACE_SECS = 1.5
# A final update that arrives this long after a fallback turn still corrects it.
LATE_UPDATE_SECS = 5.0
MAX_BUFFERED_EMBEDDINGS = 20
MAX_BUFFERED_SPEAKERS = 32
MAX_ID_LEN = 64
MAX_NAME_LEN = 128
MAX_PRESENT_PEOPLE = 16
SPEAKER_TAG_PREFIX = "[speaker: "
PRESENCE_TAG_PREFIX = "[presence: "
UNKNOWN_SPEAKER_LABEL = "unknown guest"
# A greeting needs this much silence after the last user or bot activity ...
GREETING_QUIET_SECS = 1.5
# ... and is dropped when the conversation does not go idle this soon after the person appeared.
GREETING_WINDOW_SECS = 10.0
GREETING_POLL_SECS = 0.25
# After an LLM run starts, the bot is expected to speak; do not greet meanwhile.
RESPONSE_PENDING_SECS = 15.0


@dataclass(frozen=True)
class FaceEvidence:
    """The face the client linked to the speaker of an utterance."""

    model: str
    person_id: str | None
    score: float
    link: str


@dataclass(frozen=True)
class PresentPerson:
    """One person in view, identified by face only."""

    person_id: str | None
    provisional_id: str | None
    score: float
    tier: str


@dataclass(frozen=True)
class Presence:
    """One validated ``presence-update`` message: everybody in view."""

    model: str
    people: tuple[PresentPerson, ...]
    received_at: float


@dataclass(frozen=True)
class SpeakerUpdate:
    """One validated ``speaker-update`` message."""

    model: str
    utterance_id: int
    person_id: str | None
    provisional_id: str | None
    score: float
    tier: str
    speech_ms: int
    final: bool
    embedding: np.ndarray | None
    received_at: float
    face: FaceEvidence | None = None
    face_embedding: np.ndarray | None = None


@dataclass(frozen=True)
class Speaker:
    """Who a turn is attributed to."""

    person_id: str | None = None
    provisional_id: str | None = None
    tier: str = TIER_NONE
    score: float = 0.0
    model: str = ""
    # Model key of the face linked to the speaker, when there was one.
    face_model: str = ""
    # True when no fresh estimate was available and the previous speaker was reused.
    fallback: bool = False

    @property
    def trusted_person_id(self) -> str | None:
        """The person to apply per-person policies for, or None for guest policies.

        Only a confident match (tier ``high`` or ``verified``) is trusted;
        low-confidence matches, fallbacks and unknown speakers are guests.
        """
        return self.person_id if self.tier in TRUSTED_TIERS else None

    @property
    def verified(self) -> bool:
        """Whether the voice was confirmed by the speaker's face, for policies that require both."""
        return self.tier == TIER_VERIFIED and self.person_id is not None

    @property
    def is_guest(self) -> bool:
        """Whether guest policies apply to this speaker."""
        return self.trusted_person_id is None


GUEST = Speaker()


def _clean_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if 0 < len(value) <= MAX_ID_LEN else None


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    value = float(value)
    return value if value == value and abs(value) != float("inf") else default


def _score(value: object) -> float:
    return max(-1.0, min(1.0, _number(value)))


def _parse_face(value: object) -> FaceEvidence | None:
    from monitoring.voice_id import valid_model_key

    if not isinstance(value, dict) or not valid_model_key(value.get("model")) or value.get("link") not in FACE_LINKS:
        return None
    return FaceEvidence(
        model=value["model"].strip(),
        person_id=_clean_id(value.get("person_id")),
        score=_score(value.get("score")),
        link=value["link"],
    )


def parse_speaker_update(payload: object, *, now: float | None = None) -> SpeakerUpdate | None:
    """Validate an untrusted ``speaker-update`` payload; None when it is unusable."""
    from monitoring.voice_id import decode_vector, valid_model_key

    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    if not valid_model_key(model):
        return None
    person_id = _clean_id(payload.get("person_id"))
    provisional_id = _clean_id(payload.get("provisional_id"))
    tier = payload.get("tier")
    if tier not in TIERS:
        tier = TIER_NONE
    # Keep the tier consistent with the ids: a match needs a person, unknown needs a provisional id.
    if tier == TIER_UNKNOWN:
        person_id = None
        if not provisional_id:
            tier = TIER_NONE
    elif person_id:
        provisional_id = None
    elif tier in (TIER_VERIFIED, TIER_HIGH, TIER_LOW):
        tier = TIER_UNKNOWN if provisional_id else TIER_NONE
    face = _parse_face(payload.get("face"))
    if tier == TIER_VERIFIED and (face is None or face.person_id != person_id):
        # ``verified`` needs the linked face to name the same person as the voice.
        tier = TIER_HIGH
    final = payload.get("final") is True
    return SpeakerUpdate(
        model=model.strip(),
        utterance_id=int(_number(payload.get("utterance_id"), -1)),
        person_id=person_id,
        provisional_id=provisional_id,
        score=_score(payload.get("score")),
        tier=tier,
        speech_ms=max(0, int(_number(payload.get("speech_ms")))),
        final=final,
        embedding=decode_vector(payload.get("embedding")) if final else None,
        received_at=time.monotonic() if now is None else now,
        face=face,
        face_embedding=decode_vector(payload.get("face_embedding")) if final and face is not None else None,
    )


def parse_presence_update(payload: object, *, now: float | None = None) -> Presence | None:
    """Validate an untrusted ``presence-update`` payload; None when it is unusable."""
    from monitoring.voice_id import valid_model_key

    if not isinstance(payload, dict) or not valid_model_key(payload.get("model")):
        return None
    entries = payload.get("people")
    if not isinstance(entries, list):
        return None
    people: list[PresentPerson] = []
    seen: set[str] = set()
    for entry in entries[:MAX_PRESENT_PEOPLE]:
        if not isinstance(entry, dict):
            continue
        person_id = _clean_id(entry.get("person_id"))
        provisional_id = _clean_id(entry.get("provisional_id"))
        tier = entry.get("tier")
        # Same consistency rule as for speakers: a match needs a person, anybody else is unknown.
        if tier in (TIER_HIGH, TIER_LOW) and person_id:
            provisional_id = None
        else:
            person_id, tier = None, TIER_UNKNOWN
        key = person_id or provisional_id
        if key:
            if key in seen:
                continue
            seen.add(key)
        people.append(PresentPerson(person_id, provisional_id, _score(entry.get("score")), tier))
    return Presence(
        model=payload["model"].strip(),
        people=tuple(people),
        received_at=time.monotonic() if now is None else now,
    )


class VoiceIdTracker:
    """Session state of voice ID: latest estimate, sticky speaker, enrollment buffers."""

    def __init__(self, initial_person_id: str | None = None):
        """Start with nobody, or with the person picked in the client."""
        self.active = False
        self.model = ""
        # Set once the client reports faces (a linked face or a presence update).
        self.face_active = False
        self.face_model = ""
        self.presence: Presence | None = None
        self._latest: SpeakerUpdate | None = None
        self._current = Speaker(person_id=initial_person_id, tier=TIER_HIGH) if initial_person_id else GUEST
        self._turn_open = False
        self._turn_started_at = 0.0
        self._turn_seq = 0
        self._committed_at: float | None = None
        self._consumed_utterance: tuple[str, int] | None = None
        # speaker key (person id or provisional id) -> (turn_seq, model, vector)
        self._buffers: OrderedDict[str, deque[tuple[int, str, Any]]] = OrderedDict()
        self._face_buffers: OrderedDict[str, deque[tuple[int, str, Any]]] = OrderedDict()
        # provisional id -> person id, until the client relabels its cluster.
        self._aliases: dict[str, str] = {}

    # ----------------------------------------------------------------- input
    def on_update(self, update: SpeakerUpdate) -> None:
        """Keep the latest estimate and buffer final embeddings for enrollment."""
        alias = self._aliases.get(update.provisional_id or "")
        if alias and not update.person_id:
            # Enrolled a moment ago; the client has not relabelled its cluster yet.
            update = replace(update, person_id=alias, provisional_id=None, tier=TIER_LOW)
        self.active = True
        self.model = update.model
        self._latest = update
        key = update.person_id or update.provisional_id
        if update.final and update.embedding is not None and key:
            self._buffer(self._buffers, key, update.model, update.embedding)
        face = update.face
        if face is None:
            return
        self.face_active = True
        self.face_model = face.model
        # A face recognised as somebody else than the attributed speaker is not theirs to enroll.
        if update.face_embedding is not None and key and face.person_id in (None, update.person_id):
            self._buffer(self._face_buffers, key, face.model, update.face_embedding)

    def _buffer(self, buffers: OrderedDict[str, deque[tuple[int, str, Any]]], key: str, model: str, vector: Any):
        buffer = buffers.get(key)
        if buffer is None:
            buffer = buffers[key] = deque(maxlen=MAX_BUFFERED_EMBEDDINGS)
            while len(buffers) > MAX_BUFFERED_SPEAKERS:
                buffers.popitem(last=False)
        buffers.move_to_end(key)
        buffer.append((self._turn_seq, model, vector))

    def on_presence(self, presence: Presence) -> None:
        """Keep the latest set of people in view."""
        self.active = True
        self.face_active = True
        self.face_model = presence.model
        self.presence = presence

    def others_in_view(self, speaker: Speaker) -> list[PresentPerson]:
        """People in view other than ``speaker``, with just-enrolled provisional ids resolved."""
        if self.presence is None:
            return []
        others: list[PresentPerson] = []
        for present in self.presence.people:
            alias = self._aliases.get(present.provisional_id or "")
            if alias and not present.person_id:
                present = replace(present, person_id=alias, provisional_id=None, tier=TIER_LOW)
            if present.person_id and present.person_id == speaker.person_id:
                continue
            if present.provisional_id and present.provisional_id == speaker.provisional_id:
                continue
            others.append(present)
        return others

    def in_view(self, person_id: str) -> bool:
        """Whether ``person_id`` is confidently in view right now."""
        if self.presence is None:
            return False
        return any(p.person_id == person_id and p.tier == TIER_HIGH for p in self.presence.people)

    # ----------------------------------------------------------------- turns
    @property
    def turn_open(self) -> bool:
        """Whether a user turn started and is not committed yet."""
        return self._turn_open

    def start_turn(self, now: float | None = None) -> None:
        """A user turn started (the user aggregator's ``on_user_turn_started``)."""
        if not self._turn_open:
            self._turn_seq += 1
        self._turn_open = True
        self._turn_started_at = time.monotonic() if now is None else now
        self._committed_at = None

    def _fresh_update(self) -> SpeakerUpdate | None:
        update = self._latest
        if update is None or update.tier == TIER_NONE:
            return None
        if update.received_at >= self._turn_started_at:
            return update
        # Slightly early is fine, unless that utterance already decided the previous turn.
        if update.received_at < self._turn_started_at - FRESH_GRACE_SECS:
            return None
        if (update.model, update.utterance_id) == self._consumed_utterance:
            return None
        return update

    @staticmethod
    def _from_update(update: SpeakerUpdate) -> Speaker:
        return Speaker(
            person_id=update.person_id,
            provisional_id=update.provisional_id,
            tier=update.tier,
            score=update.score,
            model=update.model,
            face_model=update.face.model if update.face else "",
        )

    def commit_turn(self, now: float | None = None) -> Speaker:
        """Attribute the open turn: the fresh estimate, else the previous speaker at tier ``low``."""
        update = self._fresh_update()
        if update is not None:
            speaker = self._from_update(update)
            self._consumed_utterance = (update.model, update.utterance_id)
        else:
            previous = self._current
            tier = TIER_LOW if previous.tier in (*TRUSTED_TIERS, TIER_LOW) else previous.tier
            speaker = replace(previous, tier=tier, fallback=True)
        self._current = speaker
        self._turn_open = False
        self._committed_at = time.monotonic() if now is None else now
        return speaker

    def late_correction(self, update: SpeakerUpdate) -> Speaker | None:
        """Speaker of a final update that arrived just after a fallback commit, else None."""
        if self._turn_open or self._committed_at is None or not self._current.fallback:
            return None
        if not update.final or update.tier == TIER_NONE:
            return None
        if update.received_at - self._committed_at > LATE_UPDATE_SECS:
            return None
        alias = self._aliases.get(update.provisional_id or "")
        if alias and not update.person_id:
            update = replace(update, person_id=alias, provisional_id=None, tier=TIER_LOW)
        self._current = self._from_update(update)
        return self._current

    def current_speaker(self) -> Speaker:
        """Speaker of the latest committed turn (``GUEST`` before the first one).

        This is the single entry point for a policy layer: use
        ``current_speaker().trusted_person_id`` and fall back to guest policies
        when it is None.
        """
        return self._current

    def set_current(self, speaker: Speaker) -> None:
        """Override the current speaker (unknown person id, enrollment)."""
        self._current = speaker

    # ------------------------------------------------------------ enrollment
    def _sample(
        self, buffers: OrderedDict[str, deque[tuple[int, str, Any]]], wanted_model: str, latest_turn_only: bool
    ) -> list[Any]:
        speaker = self._current
        key = speaker.person_id or speaker.provisional_id
        if not key or key not in buffers:
            return []
        return [
            vector
            for turn_seq, model, vector in buffers[key]
            if model == wanted_model and (not latest_turn_only or turn_seq == self._turn_seq)
        ]

    def enrollment_sample(self, *, latest_turn_only: bool = False) -> list[Any]:
        """Buffered final voice embeddings of the current speaker, for the current model."""
        return self._sample(self._buffers, self.model, latest_turn_only)

    def face_enrollment_sample(self, *, latest_turn_only: bool = False) -> list[Any]:
        """Buffered face embeddings linked to the current speaker, for the current face model."""
        return self._sample(self._face_buffers, self.face_model, latest_turn_only)

    def mark_enrolled(self, person_id: str) -> str | None:
        """The current speaker is now ``person_id``; returns the provisional id it replaces."""
        speaker = self._current
        provisional_id = speaker.provisional_id
        # The buffered sample is stored now; drop it so a later enrollment does not store it twice.
        self._buffers.pop(speaker.person_id or provisional_id or "", None)
        self._face_buffers.pop(speaker.person_id or provisional_id or "", None)
        if provisional_id:
            self._aliases[provisional_id] = person_id
        self._current = Speaker(person_id=person_id, tier=TIER_HIGH, score=speaker.score, model=self.model)
        return provisional_id


def enroll_tool_schema() -> FunctionSchema:
    """LLM schema of the server-side ``enroll_speaker`` tool."""
    return FunctionSchema(
        name=ENROLL_TOOL_NAME,
        description=(
            "Remember the voice of the person who is speaking right now (and their face, when the camera "
            "sees them) so you recognise them next time. Call it only after the person told you their name "
            "and agreed that you remember them this way."
        ),
        properties={"name": {"type": "string", "description": "The person's name, as they said it."}},
        required=["name"],
    )


def with_enroll_tool(tools: object) -> ToolsSchema:
    """``tools`` (a ``ToolsSchema`` or not given) plus the ``enroll_speaker`` tool."""
    if isinstance(tools, ToolsSchema):
        standard = [t for t in tools.standard_tools if getattr(t, "name", None) != ENROLL_TOOL_NAME]
        return ToolsSchema(standard_tools=[*standard, enroll_tool_schema()], custom_tools=tools.custom_tools)
    return ToolsSchema(standard_tools=[enroll_tool_schema()])


def speaker_label(speaker: Speaker, name: str | None) -> str:
    """How a speaker is named to the LLM."""
    if not name:
        return UNKNOWN_SPEAKER_LABEL
    return name if speaker.tier in TRUSTED_TIERS else f"{name} (uncertain)"


def in_view_label(names: list[str], unknown: int) -> str:
    """How the other people in view are listed in the speaker tag ("" when there is nobody)."""
    parts = list(names)
    if unknown == 1:
        parts.append(UNKNOWN_SPEAKER_LABEL)
    elif unknown > 1:
        parts.append(f"{unknown} unknown guests")
    return f"; also in view: {', '.join(parts)}" if parts else ""


def tag_user_message(message: object, label: str) -> bool:
    """Prefix a user context message with the speaker tag; False when it is not taggable."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    tag = f"{SPEAKER_TAG_PREFIX}{label}]"
    content = message.get("content")
    if isinstance(content, str):
        if content.startswith(SPEAKER_TAG_PREFIX):
            return False
        message["content"] = f"{tag} {content}"
        return True
    if isinstance(content, list):
        first = content[0] if content else None
        if isinstance(first, dict) and str(first.get("text", "")).startswith(SPEAKER_TAG_PREFIX):
            return False
        message["content"] = [{"type": "text", "text": tag}, *content]
        return True
    return False


def _find_target(name: str) -> dict[str, Any] | None:
    from monitoring import memories

    store = memories.live_store()
    return memories.find_person_by_name(store, name) if store is not None else None


def _bind_voice(
    name: str,
    model: str,
    vectors: list[Any],
    session_id: str | None,
    face_model: str = "",
    face_vectors: list[Any] | None = None,
) -> dict[str, Any] | None:
    from monitoring import memories, voice_id

    store = memories.live_store()
    if store is None:
        return None
    person = memories.find_or_create_person(store, name)
    kept = voice_id.add_embeddings(store, person["id"], model, vectors, source="live:enroll", session_id=session_id)
    entry = voice_id.centroids(store, model, person_id=person["id"]).get(person["id"])
    if not kept or entry is None:
        return None
    bound = {
        "person": person,
        "centroid": voice_id.encode_vector(entry["centroid"]),
        "count": entry["count"],
    }
    # The face is optional: the voice alone is a valid enrollment.
    if face_model and face_vectors:
        kept = voice_id.add_embeddings(
            store, person["id"], face_model, face_vectors, source="live:enroll", session_id=session_id
        )
        face = voice_id.centroids(store, face_model, person_id=person["id"]).get(person["id"])
        if kept and face is not None:
            bound["face"] = {
                "model": face_model,
                "centroid": voice_id.encode_vector(face["centroid"]),
                "count": face["count"],
            }
    return bound


def voice_id_available() -> bool:
    """Whether the people store is reachable (voice ID needs it for names and enrollment)."""
    try:
        from monitoring import memories

        return memories.live_store() is not None
    except Exception as exc:
        logger.opt(exception=exc).warning("Voice ID disabled: the people store is unavailable")
        return False


class VoiceIdSession:
    """Voice and face ID for one live session: turn attribution, memory swap, presence and enrollment."""

    def __init__(
        self,
        *,
        context: LLMContext,
        pinned_index: int,
        render_pinned: Callable[[PersonContext | None, bool, bool], str],
        queue_frame: Callable[[Frame], Awaitable[None]],
        session_id: Callable[[], str | None],
        turn_idx: Callable[[], int],
        initial_person: PersonContext | None = None,
    ):
        """Bind voice ID to the session context.

        Args:
            context: The shared LLM context of the session.
            pinned_index: Index of the context message that carries the prompt catalog content.
            render_pinned: Renders that message for ``(memory person, voice ID active, face ID active)``.
            queue_frame: Queues a frame on the pipeline (``speaker-enrolled``, the greeting run).
            session_id: Returns the recorded session id, or None when not recording.
            turn_idx: Returns the recorder's current turn index (``-1`` when not recording).
            initial_person: The person picked in the client, if any.
        """
        self.tracker = VoiceIdTracker(initial_person.person["id"] if initial_person else None)
        self._context = context
        self._pinned_index = pinned_index
        self._render_pinned = render_pinned
        self._queue_frame = queue_frame
        self._session_id = session_id
        self._turn_idx = turn_idx
        self._memory_person = initial_person
        self._people: dict[str, PersonContext | None] = {}
        if initial_person:
            self._people[initial_person.person["id"]] = initial_person
        self._last_turn_idx = -1
        self._background: set[asyncio.Task] = set()
        # Conversation activity, to greet somebody only while nobody talks.
        self._user_speaking = False
        self._bot_speaking = False
        self._response_pending_until = 0.0
        self._last_activity_at = time.monotonic()
        # People who were greeted, spoke, or missed their greeting window: never greeted (again).
        self._greeted: set[str] = {initial_person.person["id"]} if initial_person else set()
        self._greeting_deadlines: dict[str, float] = {}
        self._greeter: asyncio.Task | None = None

    # ---------------------------------------------------------------- policy
    def current_speaker(self) -> Speaker:
        """Speaker of the latest committed turn; see ``VoiceIdTracker.current_speaker``."""
        return self.tracker.current_speaker()

    # ------------------------------------------------------------- internals
    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _person(self, person_id: str | None, *, refresh: bool = False) -> PersonContext | None:
        if not person_id:
            return None
        if refresh or person_id not in self._people:
            self._people[person_id] = await load_person_context(person_id)
        return self._people[person_id]

    def _render(self) -> None:
        messages = self._context.get_messages()
        if not 0 <= self._pinned_index < len(messages):
            return
        message = messages[self._pinned_index]
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = self._render_pinned(self._memory_person, self.tracker.active, self.tracker.face_active)

    def _set_memory_person(self, person: PersonContext | None) -> None:
        current_id = self._memory_person.person["id"] if self._memory_person else None
        new_id = person.person["id"] if person else None
        if current_id == new_id and person is self._memory_person:
            return
        self._memory_person = person
        self._render()
        if person is not None and current_id != new_id:
            logger.info(f"Voice ID: now talking with {person.person['name']} ({len(person.memories)} memories)")
            self._spawn(record_memory_uses(self._session_id(), person))
        elif person is None:
            logger.info("Voice ID: speaker changed to an unknown guest; person memories removed")

    def _source(self, speaker: Speaker) -> str:
        source = f"live:voice-id:{speaker.model or self.tracker.model}"
        return source if speaker.tier == TIER_HIGH else f"{source}:{speaker.tier}"

    def _on_tracker_change(self, was_active: bool, was_face_active: bool) -> None:
        if not was_active:
            self._activate()
        elif self.tracker.face_active and not was_face_active:
            self._render()
            logger.info(f"Face ID active (model={self.tracker.face_model})")

    def _record_turn(self, speaker: Speaker, turn_idx: int) -> None:
        if turn_idx < 0:
            return
        self._spawn(record_turn_speaker(self._session_id(), turn_idx, speaker.person_id, self._source(speaker)))

    def _activate(self) -> None:
        """First update of the session: offer the enrollment tool and explain speaker tags."""
        self._context.set_tools(with_enroll_tool(self._context.tools))
        self._render()
        logger.info(f"Voice ID active (model={self.tracker.model})")

    # --------------------------------------------------------------- updates
    async def on_speaker_update(self, payload: object) -> None:
        """Handle a ``speaker-update`` client message."""
        update = parse_speaker_update(payload)
        if update is None:
            logger.warning("Ignoring malformed speaker-update message")
            return
        was_active, was_face_active = self.tracker.active, self.tracker.face_active
        self.tracker.on_update(update)
        self._on_tracker_change(was_active, was_face_active)
        corrected = self.tracker.late_correction(update)
        if corrected is not None:
            person = await self._person(corrected.person_id)
            if corrected.person_id and person is None:
                self.tracker.set_current(replace(corrected, person_id=None, tier=TIER_UNKNOWN))
                return
            self._record_turn(corrected, self._last_turn_idx)

    def on_user_turn_started(self) -> None:
        """The user started a turn."""
        self.tracker.start_turn()
        self._last_activity_at = time.monotonic()

    # -------------------------------------------------------------- presence
    async def on_presence_update(self, payload: object) -> None:
        """Handle a ``presence-update`` client message."""
        presence = parse_presence_update(payload)
        if presence is None:
            logger.warning("Ignoring malformed presence-update message")
            return
        was_active, was_face_active = self.tracker.active, self.tracker.face_active
        self.tracker.on_presence(presence)
        self._on_tracker_change(was_active, was_face_active)
        deadline = time.monotonic() + GREETING_WINDOW_SECS
        for present in presence.people:
            if present.tier == TIER_HIGH and present.person_id and present.person_id not in self._greeted:
                # First time in view this session: one chance to be greeted, taken or not.
                self._greeted.add(present.person_id)
                self._greeting_deadlines[present.person_id] = deadline
        if self._greeting_deadlines and (self._greeter is None or self._greeter.done()):
            self._greeter = asyncio.ensure_future(self._greet_when_idle())
            self._background.add(self._greeter)
            self._greeter.add_done_callback(self._background.discard)

    def on_activity(self, frame: Frame) -> None:
        """Track who is talking (frames seen by ``SpeakerTurnProcessor``)."""
        now = time.monotonic()
        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._response_pending_until = 0.0
        elif isinstance(frame, LLMContextFrame):
            self._response_pending_until = now + RESPONSE_PENDING_SECS
        else:
            return
        self._last_activity_at = now

    def conversation_idle(self, now: float | None = None) -> bool:
        """Whether nobody is talking and no reply is expected, for long enough to speak first."""
        now = time.monotonic() if now is None else now
        if self.tracker.turn_open or self._user_speaking or self._bot_speaking:
            return False
        if now < self._response_pending_until:
            return False
        return now - self._last_activity_at >= GREETING_QUIET_SECS

    async def _greet_when_idle(self) -> None:
        """Greet the people who just came into view, once the conversation is idle."""
        while self._greeting_deadlines:
            now = time.monotonic()
            for person_id, deadline in list(self._greeting_deadlines.items()):
                if now >= deadline or not self.tracker.in_view(person_id):
                    del self._greeting_deadlines[person_id]
            if not self._greeting_deadlines:
                return
            if not self.conversation_idle(now):
                await asyncio.sleep(GREETING_POLL_SECS)
                continue
            person_ids = list(self._greeting_deadlines)
            self._greeting_deadlines.clear()
            names = [p.person["name"] for p in [await self._person(pid) for pid in person_ids] if p is not None]
            # Loading names awaited: somebody may have started talking meanwhile.
            if not names or not self.conversation_idle():
                return
            self._context.add_message(
                {"role": "user", "content": f"{PRESENCE_TAG_PREFIX}{', '.join(names)} came into view]"}
            )
            self._response_pending_until = time.monotonic() + RESPONSE_PENDING_SECS
            logger.info(f"Face ID: greeting {', '.join(names)}")
            await self._queue_frame(LLMRunFrame())

    async def commit_turn(self, context: LLMContext) -> None:
        """Attribute the open user turn, tag its message and swap the person memories."""
        if not self.tracker.active or not self.tracker.turn_open:
            return
        speaker = self.tracker.commit_turn()
        # The LLM answers this turn next; nobody is greeted until it has spoken.
        self._response_pending_until = time.monotonic() + RESPONSE_PENDING_SECS
        person = await self._person(speaker.person_id)
        if speaker.person_id and person is None:
            # Unknown or archived id (stale client gallery): a guest.
            speaker = replace(speaker, person_id=None, tier=TIER_UNKNOWN)
            self.tracker.set_current(speaker)

        if speaker.person_id:
            # Somebody who is already talking is not greeted on top of it.
            self._greeted.add(speaker.person_id)
            self._greeting_deadlines.pop(speaker.person_id, None)

        messages = context.get_messages()
        if messages:
            label = speaker_label(speaker, person.person["name"] if person else None)
            tag_user_message(messages[-1], label + await self._in_view_label(speaker))

        # Memories follow confident identifications only: a fallback or uncertain
        # turn of the same person keeps them, anybody else removes them.
        memory_id = self._memory_person.person["id"] if self._memory_person else None
        if person is not None and speaker.tier in TRUSTED_TIERS:
            self._set_memory_person(person)
        elif speaker.person_id != memory_id:
            self._set_memory_person(None)

        self._last_turn_idx = self._turn_idx()
        self._record_turn(speaker, self._last_turn_idx)

    async def _in_view_label(self, speaker: Speaker) -> str:
        names: list[str] = []
        unknown = 0
        for present in self.tracker.others_in_view(speaker):
            person = await self._person(present.person_id)
            if person is None:
                unknown += 1
            elif present.tier == TIER_HIGH:
                names.append(person.person["name"])
            else:
                names.append(f"{person.person['name']} (uncertain)")
        return in_view_label(names, unknown)

    # ------------------------------------------------------------ enrollment
    async def enroll_speaker(self, params: FunctionCallParams) -> None:
        """Server-side ``enroll_speaker(name)`` tool handler."""
        result = await self._enroll(str(params.arguments.get("name") or ""))
        await params.result_callback(result)

    async def _enroll(self, name: str) -> dict[str, Any]:
        name = " ".join(name.split())
        if not name or len(name) > MAX_NAME_LEN:
            return {"error": "Ask the person for their name first."}
        speaker = self.tracker.current_speaker()
        no_sample = {"error": "No usable voice sample yet. Ask the person to say a full sentence, then try again."}
        if not self.tracker.active:
            return no_sample
        try:
            target = await asyncio.to_thread(_find_target, name)
            same_person = bool(speaker.person_id) and target is not None and target["id"] == speaker.person_id
            if speaker.person_id and not same_person:
                if speaker.tier in TRUSTED_TIERS:
                    known = await self._person(speaker.person_id)
                    known_name = known.person["name"] if known else "someone else"
                    return {"error": f"This voice is already recognised as {known_name}; it was not changed."}
                # An uncertain match under another name: trust only what was just said.
                latest_turn_only = True
            else:
                latest_turn_only = False
            vectors = self.tracker.enrollment_sample(latest_turn_only=latest_turn_only)
            if not vectors:
                return no_sample
            model = self.tracker.model
            face_model = self.tracker.face_model
            face_vectors = self.tracker.face_enrollment_sample(latest_turn_only=latest_turn_only)
            bound = await asyncio.to_thread(
                _bind_voice, name, model, vectors, self._session_id(), face_model, face_vectors
            )
        except Exception as exc:
            logger.opt(exception=exc).warning("Voice enrollment failed")
            return {"error": "The voice could not be saved."}
        if bound is None:
            return no_sample

        person = bound["person"]
        provisional_id = self.tracker.mark_enrolled(person["id"])
        enrolled = self.tracker.current_speaker()
        self._set_memory_person(await self._person(person["id"], refresh=True))
        self._record_turn(enrolled, self._last_turn_idx)
        data = {
            "type": SPEAKER_ENROLLED_MESSAGE,
            "model": model,
            "provisional_id": provisional_id,
            "person_id": person["id"],
            "name": person["name"],
            "centroid": bound["centroid"],
            "count": bound["count"],
        }
        if "face" in bound:
            data["face"] = bound["face"]
        await self._queue_frame(RTVIServerMessageFrame(data=data))
        logger.info(
            f"Voice ID: enrolled {person['name']} ({bound['count']} voice embeddings, model={model}"
            + (f"; {bound['face']['count']} face embeddings, model={face_model})" if "face" in bound else ")")
        )
        return {"status": "enrolled", "name": person["name"]}


class SpeakerTurnProcessor(FrameProcessor):
    """Commit the speaker of a user turn right before its context reaches the LLM.

    Sits between the user aggregator and the LLM. Unlike the per-turn language
    reminder, the speaker tag is written into the shared context, so later
    requests (tool follow-ups, the next turns, summaries) still know who said what.
    """

    def __init__(self, session: VoiceIdSession):
        """Build the processor for one session."""
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Forward frames; attribute the user turn when its context frame passes."""
        await super().process_frame(frame, direction)
        self._session.on_activity(frame)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            try:
                await self._session.commit_turn(frame.context)
            except Exception as exc:
                logger.opt(exception=exc).warning("Voice ID could not attribute the turn; continuing")
        await self.push_frame(frame, direction)

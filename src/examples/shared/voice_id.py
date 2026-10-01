# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live side of voice ID: who is speaking in each user turn.

The client computes speaker embeddings and streams ``speaker-update`` messages
(see ``docs/voice-id-protocol.md``). This module keeps the latest estimate,
attributes each user turn to a person when the turn is committed, swaps the
per-person memory block of the pinned prompt, and enrolls unknown speakers
through the server-side ``enroll_speaker`` tool.

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
from pipecat.frames.frames import Frame, LLMContextFrame
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
SPEAKER_ENROLLED_MESSAGE = "speaker-enrolled"
VOICE_ID_ADDON_KEY = "voice_id_addon"
ENROLL_TOOL_NAME = "enroll_speaker"

TIER_HIGH = "high"
TIER_LOW = "low"
TIER_UNKNOWN = "unknown"
TIER_NONE = "none"
TIERS = (TIER_HIGH, TIER_LOW, TIER_UNKNOWN, TIER_NONE)

# The client's VAD and data channel are faster than the server's turn start, so
# an update that arrives slightly before the turn opens still belongs to it.
FRESH_GRACE_SECS = 1.5
# A final update that arrives this long after a fallback turn still corrects it.
LATE_UPDATE_SECS = 5.0
MAX_BUFFERED_EMBEDDINGS = 20
MAX_BUFFERED_SPEAKERS = 32
MAX_ID_LEN = 64
MAX_NAME_LEN = 128
SPEAKER_TAG_PREFIX = "[speaker: "
UNKNOWN_SPEAKER_LABEL = "unknown guest"


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


@dataclass(frozen=True)
class Speaker:
    """Who a turn is attributed to."""

    person_id: str | None = None
    provisional_id: str | None = None
    tier: str = TIER_NONE
    score: float = 0.0
    model: str = ""
    # True when no fresh estimate was available and the previous speaker was reused.
    fallback: bool = False

    @property
    def trusted_person_id(self) -> str | None:
        """The person to apply per-person policies for, or None for guest policies.

        Only a confident match (tier ``high``) is trusted; low-confidence matches,
        fallbacks and unknown speakers are guests.
        """
        return self.person_id if self.tier == TIER_HIGH else None

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
    elif tier in (TIER_HIGH, TIER_LOW):
        tier = TIER_UNKNOWN if provisional_id else TIER_NONE
    final = payload.get("final") is True
    return SpeakerUpdate(
        model=model.strip(),
        utterance_id=int(_number(payload.get("utterance_id"), -1)),
        person_id=person_id,
        provisional_id=provisional_id,
        score=max(-1.0, min(1.0, _number(payload.get("score")))),
        tier=tier,
        speech_ms=max(0, int(_number(payload.get("speech_ms")))),
        final=final,
        embedding=decode_vector(payload.get("embedding")) if final else None,
        received_at=time.monotonic() if now is None else now,
    )


class VoiceIdTracker:
    """Session state of voice ID: latest estimate, sticky speaker, enrollment buffers."""

    def __init__(self, initial_person_id: str | None = None):
        """Start with nobody, or with the person picked in the client."""
        self.active = False
        self.model = ""
        self._latest: SpeakerUpdate | None = None
        self._current = Speaker(person_id=initial_person_id, tier=TIER_HIGH) if initial_person_id else GUEST
        self._turn_open = False
        self._turn_started_at = 0.0
        self._turn_seq = 0
        self._committed_at: float | None = None
        self._consumed_utterance: tuple[str, int] | None = None
        # speaker key (person id or provisional id) -> (turn_seq, model, vector)
        self._buffers: OrderedDict[str, deque[tuple[int, str, Any]]] = OrderedDict()
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
            buffer = self._buffers.get(key)
            if buffer is None:
                buffer = self._buffers[key] = deque(maxlen=MAX_BUFFERED_EMBEDDINGS)
                while len(self._buffers) > MAX_BUFFERED_SPEAKERS:
                    self._buffers.popitem(last=False)
            self._buffers.move_to_end(key)
            buffer.append((self._turn_seq, update.model, update.embedding))

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
        )

    def commit_turn(self, now: float | None = None) -> Speaker:
        """Attribute the open turn: the fresh estimate, else the previous speaker at tier ``low``."""
        update = self._fresh_update()
        if update is not None:
            speaker = self._from_update(update)
            self._consumed_utterance = (update.model, update.utterance_id)
        else:
            previous = self._current
            tier = TIER_LOW if previous.tier in (TIER_HIGH, TIER_LOW) else previous.tier
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
    def enrollment_sample(self, *, latest_turn_only: bool = False) -> list[Any]:
        """Buffered final embeddings of the current speaker, for the current model."""
        speaker = self._current
        key = speaker.person_id or speaker.provisional_id
        if not key or key not in self._buffers:
            return []
        return [
            vector
            for turn_seq, model, vector in self._buffers[key]
            if model == self.model and (not latest_turn_only or turn_seq == self._turn_seq)
        ]

    def mark_enrolled(self, person_id: str) -> str | None:
        """The current speaker is now ``person_id``; returns the provisional id it replaces."""
        speaker = self._current
        provisional_id = speaker.provisional_id
        # The buffered sample is stored now; drop it so a later enrollment does not store it twice.
        self._buffers.pop(speaker.person_id or provisional_id or "", None)
        if provisional_id:
            self._aliases[provisional_id] = person_id
        self._current = Speaker(person_id=person_id, tier=TIER_HIGH, score=speaker.score, model=self.model)
        return provisional_id


def enroll_tool_schema() -> FunctionSchema:
    """LLM schema of the server-side ``enroll_speaker`` tool."""
    return FunctionSchema(
        name=ENROLL_TOOL_NAME,
        description=(
            "Remember the voice of the person who is speaking right now so you recognise them next time. "
            "Call it only after the person told you their name and agreed that you remember their voice."
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
    return name if speaker.tier == TIER_HIGH else f"{name} (uncertain)"


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


def _bind_voice(name: str, model: str, vectors: list[Any], session_id: str | None) -> dict[str, Any] | None:
    from monitoring import memories, voice_id

    store = memories.live_store()
    if store is None:
        return None
    person = memories.find_or_create_person(store, name)
    kept = voice_id.add_embeddings(store, person["id"], model, vectors, source="live:enroll", session_id=session_id)
    entry = voice_id.centroids(store, model, person_id=person["id"]).get(person["id"])
    if not kept or entry is None:
        return None
    return {
        "person": person,
        "centroid": voice_id.encode_vector(entry["centroid"]),
        "count": entry["count"],
    }


def voice_id_available() -> bool:
    """Whether the people store is reachable (voice ID needs it for names and enrollment)."""
    try:
        from monitoring import memories

        return memories.live_store() is not None
    except Exception as exc:
        logger.opt(exception=exc).warning("Voice ID disabled: the people store is unavailable")
        return False


class VoiceIdSession:
    """Voice ID for one live session: turn attribution, memory swap and enrollment."""

    def __init__(
        self,
        *,
        context: LLMContext,
        pinned_index: int,
        render_pinned: Callable[[PersonContext | None, bool], str],
        queue_frame: Callable[[Frame], Awaitable[None]],
        session_id: Callable[[], str | None],
        turn_idx: Callable[[], int],
        initial_person: PersonContext | None = None,
    ):
        """Bind voice ID to the session context.

        Args:
            context: The shared LLM context of the session.
            pinned_index: Index of the context message that carries the prompt catalog content.
            render_pinned: Renders that message for ``(memory person, voice ID active)``.
            queue_frame: Queues a frame on the pipeline (for ``speaker-enrolled``).
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
            message["content"] = self._render_pinned(self._memory_person, self.tracker.active)

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
        was_active = self.tracker.active
        self.tracker.on_update(update)
        if not was_active:
            self._activate()
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

    async def commit_turn(self, context: LLMContext) -> None:
        """Attribute the open user turn, tag its message and swap the person memories."""
        if not self.tracker.active or not self.tracker.turn_open:
            return
        speaker = self.tracker.commit_turn()
        person = await self._person(speaker.person_id)
        if speaker.person_id and person is None:
            # Unknown or archived id (stale client gallery): a guest.
            speaker = replace(speaker, person_id=None, tier=TIER_UNKNOWN)
            self.tracker.set_current(speaker)

        messages = context.get_messages()
        if messages:
            tag_user_message(messages[-1], speaker_label(speaker, person.person["name"] if person else None))

        # Memories follow confident identifications only: a fallback or uncertain
        # turn of the same person keeps them, anybody else removes them.
        memory_id = self._memory_person.person["id"] if self._memory_person else None
        if person is not None and speaker.tier == TIER_HIGH:
            self._set_memory_person(person)
        elif speaker.person_id != memory_id:
            self._set_memory_person(None)

        self._last_turn_idx = self._turn_idx()
        self._record_turn(speaker, self._last_turn_idx)

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
                if speaker.tier == TIER_HIGH:
                    known = await self._person(speaker.person_id)
                    known_name = known.person["name"] if known else "someone else"
                    return {"error": f"This voice is already recognised as {known_name}; it was not changed."}
                # An uncertain match under another name: trust only what was just said.
                vectors = self.tracker.enrollment_sample(latest_turn_only=True)
            else:
                vectors = self.tracker.enrollment_sample()
            if not vectors:
                return no_sample
            model = self.tracker.model
            bound = await asyncio.to_thread(_bind_voice, name, model, vectors, self._session_id())
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
        await self._queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": SPEAKER_ENROLLED_MESSAGE,
                    "model": model,
                    "provisional_id": provisional_id,
                    "person_id": person["id"],
                    "name": person["name"],
                    "centroid": bound["centroid"],
                    "count": bound["count"],
                }
            )
        )
        logger.info(f"Voice ID: enrolled {person['name']} ({bound['count']} embeddings, model={model})")
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
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            try:
                await self._session.commit_turn(frame.context)
            except Exception as exc:
                logger.opt(exception=exc).warning("Voice ID could not attribute the turn; continuing")
        await self.push_frame(frame, direction)

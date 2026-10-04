# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-driven settings for the intent engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from utils import PROJECT_ROOT, parse_env_bool, parse_env_float

PACKAGED_SENTENCES_DIR = Path(__file__).resolve().parent / "sentences"

# Home Assistant intents the engine may execute without the LLM. Wildcard intents
# (media search, broadcast, list items) and timers are left out on purpose: a
# wildcard slot matches almost any sentence, and timers need a satellite device.
DEFAULT_ALLOWED_INTENTS: tuple[str, ...] = (
    "HassTurnOn",
    "HassTurnOff",
    "HassLightSet",
    "HassSetPosition",
    "HassClimateSetTemperature",
    "HassClimateGetTemperature",
    "HassSetVolume",
    "HassSetVolumeRelative",
    "HassMediaPause",
    "HassMediaUnpause",
    "HassMediaNext",
    "HassMediaPrevious",
    "HassMediaPlayerMute",
    "HassMediaPlayerUnmute",
    "HassGetState",
    "HassGetCurrentTime",
    "HassGetCurrentDate",
    "HassGetWeather",
)

# ``HassGetState`` sentences are tagged with a response key. Only the single-entity
# keys are answered here: the REST response does not say which entities matched a
# yes/no, "which", or "how many" question, so those stay with the LLM.
ALLOWED_STATE_RESPONSE_KEYS: frozenset[str] = frozenset({"default", "one"})

# Entity domains the engine never controls, even when Home Assistant exposes them to
# Assist: a misheard sentence must not unlock a door or disarm an alarm.
DEFAULT_BLOCKED_DOMAINS: tuple[str, ...] = ("lock", "alarm_control_panel")


def _parse_env_list(name: str) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into a tuple of trimmed items."""
    return tuple(item.strip() for item in (os.getenv(name) or "").split(",") if item.strip())


@dataclass(frozen=True)
class IntentEngineConfig:
    """Resolved intent-engine settings for one session."""

    enabled: bool = False
    home_assistant_url: str = ""
    # Never log or snapshot the token; ``repr=False`` keeps it out of dataclass reprs.
    home_assistant_token: str = field(default="", repr=False)
    area: str = ""
    dry_run: bool = False
    timeout_secs: float = 3.0
    entity_cache_ttl_secs: float = 60.0
    wake_words: tuple[str, ...] = ()
    allowed_intents: tuple[str, ...] = DEFAULT_ALLOWED_INTENTS
    blocked_domains: tuple[str, ...] = DEFAULT_BLOCKED_DOMAINS
    sentences_dir: Path = PACKAGED_SENTENCES_DIR

    @classmethod
    def from_env(cls) -> IntentEngineConfig:
        """Read the ``INTENT_ENGINE_*`` and ``HOME_ASSISTANT_*`` variables."""
        raw_dir = (os.getenv("INTENT_ENGINE_SENTENCES_DIR") or "").strip()
        sentences_dir = PACKAGED_SENTENCES_DIR
        if raw_dir:
            sentences_dir = Path(raw_dir).expanduser()
            if not sentences_dir.is_absolute():
                sentences_dir = PROJECT_ROOT / sentences_dir
        blocked = _parse_env_list("INTENT_ENGINE_BLOCKED_DOMAINS") or DEFAULT_BLOCKED_DOMAINS
        if [item.lower() for item in blocked] == ["none"]:
            blocked = ()
        return cls(
            enabled=parse_env_bool("INTENT_ENGINE_ENABLED", False),
            home_assistant_url=(os.getenv("HOME_ASSISTANT_URL") or "").strip().rstrip("/"),
            home_assistant_token=(os.getenv("HOME_ASSISTANT_TOKEN") or "").strip(),
            area=(os.getenv("INTENT_ENGINE_AREA") or "").strip(),
            dry_run=parse_env_bool("INTENT_ENGINE_DRY_RUN", False),
            timeout_secs=parse_env_float("INTENT_ENGINE_TIMEOUT_SECS", 3.0, min_value=0.1),
            entity_cache_ttl_secs=parse_env_float("INTENT_ENGINE_ENTITY_CACHE_TTL_SECS", 60.0, min_value=0.0),
            wake_words=_parse_env_list("INTENT_ENGINE_WAKE_WORDS"),
            allowed_intents=_parse_env_list("INTENT_ENGINE_ALLOWED_INTENTS") or DEFAULT_ALLOWED_INTENTS,
            blocked_domains=tuple(item.lower() for item in blocked),
            sentences_dir=sentences_dir,
        )

    @property
    def home_assistant_configured(self) -> bool:
        """Return whether both the Home Assistant URL and token are set."""
        return bool(self.home_assistant_url and self.home_assistant_token)

    def snapshot(self) -> dict:
        """Describe the engine for the recorded session snapshot (no URL, no token)."""
        return {
            "enabled": self.enabled,
            "home_assistant": self.enabled and self.home_assistant_configured,
            "dry_run": self.dry_run,
            "area": self.area,
            "timeout_secs": self.timeout_secs,
            "wake_words": list(self.wake_words),
            "allowed_intents": list(self.allowed_intents),
            "blocked_domains": list(self.blocked_domains),
        }

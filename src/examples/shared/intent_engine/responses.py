# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the per-language response templates shipped with the sentence files."""

from __future__ import annotations

import contextlib
from datetime import date, time
from typing import Any

from jinja2 import TemplateError, meta
from jinja2.sandbox import ImmutableSandboxedEnvironment

_ENV = ImmutableSandboxedEnvironment()


def is_local_template(template: str) -> bool:
    """Return whether a template only needs ``slots``.

    Query templates read ``state`` or ``query`` and call Home Assistant template
    functions such as ``state_attr``; those are rendered by Home Assistant.
    """
    try:
        _ENV.from_string(template)  # also rejects filters Jinja does not know
        return meta.find_undeclared_variables(_ENV.parse(template)) <= {"slots"}
    except TemplateError:
        return False


def coerce_speech_slots(slots: dict[str, Any]) -> dict[str, Any]:
    """Turn the REST string forms of ``time`` and ``date`` back into objects.

    ``HassGetCurrentTime`` returns ``"11:53:44.860885"`` over REST, while its
    template reads ``slots.time.hour``. ``HassGetCurrentDate`` behaves the same.
    """
    coerced = dict(slots)
    for key, parser in (("time", time.fromisoformat), ("date", date.fromisoformat)):
        value = coerced.get(key)
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                coerced[key] = parser(value)
    return coerced


def clean_speech(text: Any) -> str:
    """Normalize whitespace the way Home Assistant does for rendered speech."""
    return " ".join(str(text or "").split())


def render_local(template: str, slots: dict[str, Any]) -> str:
    """Render a ``slots``-only template in a sandbox."""
    return clean_speech(_ENV.from_string(template).render(slots=coerce_speech_slots(slots)))

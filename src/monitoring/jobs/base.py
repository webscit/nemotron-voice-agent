# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job plug-in contract for the offline post-processing runner.

A job processes one *target* (a session id) in small resumable steps:

- call ``ctx.check_preempted()`` between steps; it raises ``Preempted`` as soon
  as a live conversation starts, so the runner can free the GPU;
- store resumable state with ``ctx.save_progress(...)`` (read it back from
  ``ctx.progress``) so a preempted job continues where it stopped;
- write results as ``annotations`` rows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from monitoring.store import ArtifactStore, SessionStore


class Preempted(Exception):
    """Raised inside a job when a live session needs the resources back."""


@dataclass
class JobContext:
    """What a running job can use."""

    job: dict[str, Any]
    store: SessionStore
    artifacts: ArtifactStore
    config: dict[str, Any]
    is_preempted: Callable[[], bool]
    progress: dict[str, Any] = field(default_factory=dict)

    def check_preempted(self) -> None:
        """Raise ``Preempted`` when the runner must yield to a live session."""
        if self.is_preempted():
            raise Preempted()

    def save_progress(self, **values: Any) -> None:
        """Persist resumable state for this job."""
        self.progress.update(values)
        self.store.update_job(self.job["id"], progress=dict(self.progress))


class Job:
    """Base class for post-processing jobs; subclasses register with ``@register``."""

    kind: ClassVar[str] = ""

    def required_services(self, ctx: JobContext) -> list[str]:
        """Containers that must be running for this job (started only while idle)."""
        return []

    def run(self, ctx: JobContext) -> None:
        """Process ``ctx.job["target"]``."""
        raise NotImplementedError


JOB_REGISTRY: dict[str, type[Job]] = {}


def register(cls: type[Job]) -> type[Job]:
    """Class decorator adding a job to the registry under ``cls.kind``."""
    JOB_REGISTRY[cls.kind] = cls
    return cls

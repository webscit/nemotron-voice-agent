# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The "dreamer": runs post-processing jobs only while no conversation is live.

Idle is decided from the session store alone (no call into the voice server),
so it keeps working unchanged once the store is a remote database:

- live   = a session with ``ended_at IS NULL`` and a heartbeat within
  ``live_stale_secs`` (the recorder heartbeats every 15 s);
- idle   = no live session and the last session activity is older than
  ``idle_grace_secs`` (avoids grabbing the GPU between two quick calls).

Jobs run one at a time; when a session opens mid-job the job raises
``Preempted`` at its next checkpoint, returns to ``pending`` (without consuming
an attempt) and every on-demand container the runner started is stopped.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from monitoring.jobs.base import JOB_REGISTRY, JobContext, Preempted
from monitoring.jobs.services import DockerServiceManager
from monitoring.store import ArtifactStore, SessionStore

DEFAULT_CONFIG_PATH = Path(__file__).with_name("dreamer.yaml")

_DEFAULTS: dict[str, Any] = {
    "poll_secs": 10.0,
    "idle_grace_secs": 60.0,
    "live_stale_secs": 60.0,
    "orphan_after_secs": 120.0,
    "max_attempts": 3,
    "auto_enqueue": ["reasr"],
    "compose_project": None,
}


def load_dreamer_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``dreamer.yaml`` (``DREAMER_CONFIG`` env var overrides the path)."""
    path = Path(path or os.getenv("DREAMER_CONFIG", "") or DEFAULT_CONFIG_PATH)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    return {**_DEFAULTS, **(raw or {})}


class Dreamer:
    """Idle-gated, preemptible job runner."""

    def __init__(
        self,
        store: SessionStore,
        artifacts: ArtifactStore,
        config: dict[str, Any],
        *,
        services: DockerServiceManager | None = None,
        clock=time.time,
    ):
        """Bind the runner to its stores; ``clock`` is injectable for tests."""
        self.store = store
        self.artifacts = artifacts
        self.config = config
        self.services = services or DockerServiceManager(project=config.get("compose_project"))
        self._clock = clock
        self._preempt_checked_at = 0.0
        self._preempted = False

    # ------------------------------------------------------------ idle state
    def live_sessions(self) -> int:
        """Number of sessions currently in progress."""
        return self.store.open_session_count(now=self._clock(), stale_after_secs=self.config["live_stale_secs"])

    def is_idle(self) -> bool:
        """True when no session is live and the grace period has elapsed."""
        if self.live_sessions():
            return False
        last = self.store.last_session_activity()
        return last is None or self._clock() - last >= self.config["idle_grace_secs"]

    def is_preempted(self) -> bool:
        """Cheap (rate-limited) check used by jobs between steps."""
        now = self._clock()
        if self._preempted or now - self._preempt_checked_at >= 1.0:
            self._preempt_checked_at = now
            self._preempted = self.live_sessions() > 0
        return self._preempted

    # ------------------------------------------------------------------ loop
    def startup(self) -> None:
        """Recover from crashes: close orphaned sessions, requeue running jobs."""
        orphans = self.store.close_orphaned_sessions(
            now=self._clock(), stale_after_secs=self.config["orphan_after_secs"]
        )
        if orphans:
            logger.info(f"Closed {len(orphans)} orphaned session(s): {', '.join(orphans)}")
        requeued = self.store.requeue_running_jobs()
        if requeued:
            logger.info(f"Requeued {requeued} interrupted job(s)")

    def enqueue_ended_sessions(self) -> int:
        """Queue the ``auto_enqueue`` jobs for every ended session (idempotent)."""
        kinds = [k for k in self.config.get("auto_enqueue") or [] if k in JOB_REGISTRY]
        added = 0
        for session_id in self.store.ended_session_ids():
            for kind in kinds:
                added += self.store.enqueue_job(kind, session_id)
        return added

    def run_once(self) -> bool:
        """Run at most one job; return whether one was attempted."""
        self.store.close_orphaned_sessions(now=self._clock(), stale_after_secs=self.config["orphan_after_secs"])
        self.enqueue_ended_sessions()
        if not self.is_idle():
            self.services.stop_started()
            return False
        job = self.store.claim_next_job(list(JOB_REGISTRY), max_attempts=self.config["max_attempts"])
        if job is None:
            self.services.stop_started()
            return False
        self._preempted = False
        self._run_job(job)
        if self._preempted:
            self.services.stop_started()
        return True

    def run_forever(self) -> None:
        """Poll forever (the ``dreamer`` compose service)."""
        self.startup()
        logger.info(f"Dreamer started (jobs: {', '.join(JOB_REGISTRY)})")
        while True:
            try:
                worked = self.run_once()
            except Exception as exc:
                logger.opt(exception=exc).error("Dreamer loop error")
                worked = False
            if not worked:
                time.sleep(self.config["poll_secs"])

    def _run_job(self, job: dict[str, Any]) -> None:
        handler = JOB_REGISTRY[job["kind"]]()
        ctx = JobContext(
            job=job,
            store=self.store,
            artifacts=self.artifacts,
            config=self.config,
            is_preempted=self.is_preempted,
            progress=dict(job.get("progress") or {}),
        )
        logger.info(f"Running job {job['kind']}#{job['id']} on {job['target']} (attempt {job['attempts']})")
        try:
            for service in handler.required_services(ctx):
                self.services.ensure_running(service, is_preempted=self.is_preempted)
                ctx.check_preempted()
            handler.run(ctx)
        except Preempted:
            logger.info(f"Job {job['kind']}#{job['id']} preempted by a live session; will resume")
            self._preempted = True
            self.store.update_job(job["id"], status="pending", attempts=job["attempts"] - 1)
            return
        except Exception as exc:
            logger.opt(exception=exc).error(f"Job {job['kind']}#{job['id']} failed")
            final = job["attempts"] >= self.config["max_attempts"]
            self.store.update_job(
                job["id"],
                status="failed" if final else "pending",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=self._clock() if final else None,
            )
            return
        self.store.update_job(job["id"], status="done", error=None, finished_at=self._clock())
        logger.info(f"Job {job['kind']}#{job['id']} done")

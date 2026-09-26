# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start/stop on-demand model containers through the Docker Engine API.

The containers are declared in compose under the ``dreamer-models`` profile and
created once (``docker compose --profile dreamer-models create``); the runner
only starts them while no conversation is live and stops them before yielding.
Talks HTTP over ``/var/run/docker.sock`` directly, so no docker CLI is needed.
"""

from __future__ import annotations

import http.client
import json
import socket
import time
from typing import Any
from urllib.parse import quote

from loguru import logger

DOCKER_SOCKET = "/var/run/docker.sock"


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


class DockerServiceManager:
    """Tracks which containers the runner started so it can stop exactly those."""

    def __init__(self, socket_path: str = DOCKER_SOCKET, *, project: str | None = None, timeout: float = 30.0):
        """Use the Docker socket at ``socket_path``; ``project`` prefixes compose service names."""
        self._socket_path = socket_path
        self._project = project
        self._timeout = timeout
        self.started: list[str] = []

    def _request(self, method: str, path: str) -> tuple[int, Any]:
        conn = _UnixHTTPConnection(self._socket_path, self._timeout)
        try:
            conn.request(method, path)
            response = conn.getresponse()
            body = response.read()
        finally:
            conn.close()
        return response.status, (json.loads(body) if body and body[:1] in (b"{", b"[") else body)

    def _container_for(self, service: str) -> str:
        """Resolve a compose service name to a container id (falls back to the raw name)."""
        filters = {"label": [f"com.docker.compose.service={service}"]}
        if self._project:
            filters["label"].append(f"com.docker.compose.project={self._project}")
        status, body = self._request("GET", f"/containers/json?all=1&filters={quote(json.dumps(filters))}")
        if status == 200 and body:
            return body[0]["Id"]
        return service

    def _state(self, container: str) -> dict[str, Any]:
        status, body = self._request("GET", f"/containers/{container}/json")
        if status != 200:
            raise RuntimeError(f"Container {container} not found (HTTP {status}); run compose create first")
        return body["State"]

    def ensure_running(self, service: str, *, wait_secs: float = 900.0, is_preempted=lambda: False) -> None:
        """Start ``service`` if needed and wait until healthy (or running without healthcheck)."""
        container = self._container_for(service)
        state = self._state(container)
        if not state.get("Running"):
            logger.info(f"Starting on-demand service {service}")
            status, body = self._request("POST", f"/containers/{container}/start")
            if status not in (204, 304):
                raise RuntimeError(f"Failed to start {service}: HTTP {status} {body!r}")
            self.started.append(service)
        deadline = time.monotonic() + wait_secs
        while time.monotonic() < deadline:
            state = self._state(container)
            health = (state.get("Health") or {}).get("Status")
            if state.get("Running") and health in (None, "healthy"):
                return
            if not state.get("Running"):
                raise RuntimeError(f"Service {service} exited (code {state.get('ExitCode')})")
            if is_preempted():
                return
            time.sleep(2.0)
        raise TimeoutError(f"Service {service} not healthy after {wait_secs:.0f}s")

    def stop_started(self) -> None:
        """Stop every container this manager started (most recent first)."""
        while self.started:
            service = self.started.pop()
            logger.info(f"Stopping on-demand service {service}")
            try:
                self._request("POST", f"/containers/{self._container_for(service)}/stop?t=10")
            except OSError as exc:
                logger.warning(f"Failed to stop {service}: {exc}")

# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Optional Pipecat Whisker debugger wiring.

Not imported by default. Enable it by pointing ``PIPECAT_SETUP_FILES`` at
this file, e.g.:

    PIPECAT_SETUP_FILES=/app/src/whisker_setup.py uv run python src/server.py

Pipecat discovers ``setup_worker_runner``/``setup_pipeline_worker`` in every
file listed there and calls them automatically (see
``pipecat.utils.startup``), so no other code change is needed.

The app (``src/server.py``) builds a brand-new ``WorkerRunner`` per client
session and tears it down when that session ends. Whisker's own README
wires ``WhiskerServer`` straight into that per-session runner via
``add_workers``, but a runner cancels every worker it owns when it
finishes -- so on a multi-session server that would stop-and-restart the
WhiskerServer (and its WebSocket listener on port 9090) on every
connect/disconnect, which races the OS releasing the port and fails the
next session with "address already in use". Instead, the WhiskerServer
gets its own runner, started once (guarded by ``_whisker_started``) and
left running for the life of the process with ``auto_end=False`` -- exactly
the pattern ``WorkerRunner.run()`` documents for "long-lived hosts that add
and remove workers over many sessions (e.g. a FastAPI server)". Each
session's pipeline worker then only attaches an observer
(``create_observer``), which is a local, runner-independent registration
and does not require sharing a bus with the WhiskerServer's runner.

Binds on all interfaces (not just localhost) so the Whisker UI can connect
from outside the container when this runs in Docker; see
https://github.com/pipecat-ai/whisker for the UI.

Also records every event to a ``.whisk`` trace file (same wire format as
the live WebSocket feed, replayable later via the UI's "Load session"),
one file per process start. Configurable via:
  - WHISKER_TRACE_PATH  (default: <project_root>/whisker_traces)
"""

import asyncio
import os
import time
from pathlib import Path

from loguru import logger
from pipecat.workers.runner import WorkerRunner
from pipecat_whisker import WhiskerServer

from utils import PROJECT_ROOT

_raw_trace_path = Path(os.getenv("WHISKER_TRACE_PATH", "whisker_traces"))
TRACE_DIR = _raw_trace_path if _raw_trace_path.is_absolute() else PROJECT_ROOT / _raw_trace_path
TRACE_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE = TRACE_DIR / f"whisker_{time.strftime('%Y%m%d_%H%M%S')}.whisk"

whisker = WhiskerServer(host="0.0.0.0", port=9090, file_name=str(TRACE_FILE))

_whisker_started = False
_whisker_lock = asyncio.Lock()


async def _ensure_whisker_started() -> None:
    """Start the WhiskerServer on its own long-lived runner, once per process."""
    global _whisker_started
    if _whisker_started:
        return
    async with _whisker_lock:
        if _whisker_started:
            return
        whisker_runner = WorkerRunner(name="whisker-runner", handle_sigint=False)
        await whisker_runner.add_workers(whisker)
        asyncio.create_task(whisker_runner.run(auto_end=False))
        _whisker_started = True
        logger.info(f"ᓚᘏᗢ Whisker trace recording to {TRACE_FILE}")


async def setup_worker_runner(runner):
    """Ensure the shared WhiskerServer is running before any session starts."""
    await _ensure_whisker_started()


async def setup_pipeline_worker(worker):
    """Attach a Whisker observer to this session's pipeline worker."""
    worker.add_observer(whisker.create_observer(worker))

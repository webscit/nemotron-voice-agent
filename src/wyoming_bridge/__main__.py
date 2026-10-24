# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Wyoming bridge: ``python -m wyoming_bridge`` (flags default to ``WYOMING_BRIDGE_*``)."""

from __future__ import annotations

import argparse
import asyncio
import os
from functools import partial
from importlib.metadata import PackageNotFoundError, version

import grpc
from loguru import logger
from wyoming.server import AsyncServer

from wyoming_bridge.handler import BridgeSettings, RivaEventHandler, build_info
from wyoming_bridge.riva_backend import DEFAULT_ASR_STOP_HISTORY_MS, RivaBackend, SpeechCatalog

_READY_TIMEOUT_SECS = 10.0


def _parse_args() -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(prog="python -m wyoming_bridge", description=__doc__)
    parser.add_argument("--uri", default=env("WYOMING_BRIDGE_URI", "tcp://0.0.0.0:10700"), help="Wyoming listen URI")
    parser.add_argument(
        "--riva-server", default=env("WYOMING_BRIDGE_RIVA_SERVER", "nemo-speech:50051"), help="Riva gRPC host:port"
    )
    parser.add_argument(
        "--language",
        default=env("WYOMING_BRIDGE_LANGUAGE", "en-US"),
        help="Locale used when a request names no language (or one the server does not list)",
    )
    parser.add_argument(
        "--tts-voice", default=env("WYOMING_BRIDGE_TTS_VOICE", ""), help="Voice used when a request names none"
    )
    parser.add_argument(
        "--tts-sample-rate",
        type=int,
        default=int(env("WYOMING_BRIDGE_TTS_SAMPLE_RATE", "22050")),
        help="Sample rate requested from Riva TTS",
    )
    parser.add_argument(
        "--asr-stop-history-ms",
        type=int,
        default=int(env("WYOMING_BRIDGE_ASR_STOP_HISTORY_MS", str(DEFAULT_ASR_STOP_HISTORY_MS))),
        help="Trailing silence before Riva ends an ASR segment (multiple of 80; -1 keeps the server setting)",
    )
    return parser.parse_args()


def _discover(backend: RivaBackend) -> SpeechCatalog:
    """Wait for the Riva server (it can take minutes to load models), then read its catalog."""
    while True:
        try:
            backend.wait_ready(_READY_TIMEOUT_SECS)
            return backend.describe()
        except (grpc.FutureTimeoutError, grpc.RpcError) as exc:
            logger.info(f"Waiting for Riva server {backend.server}: {exc.__class__.__name__}")


def _bridge_version() -> str | None:
    try:
        return version("nemotron-voice-agent")
    except PackageNotFoundError:
        return None


async def main() -> None:
    """Discover the Riva catalog and serve Wyoming clients until stopped."""
    args = _parse_args()
    backend = RivaBackend(args.riva_server, args.asr_stop_history_ms)
    catalog = await asyncio.to_thread(_discover, backend)
    logger.info(
        f"Riva {args.riva_server}: ASR models {list(catalog.asr_models)}, "
        f"{len(catalog.asr_languages)} ASR languages, TTS voices {list(catalog.tts_voices)}"
    )
    settings = BridgeSettings(language=args.language, tts_voice=args.tts_voice, tts_sample_rate=args.tts_sample_rate)
    info = build_info(catalog, _bridge_version())
    server = AsyncServer.from_uri(args.uri)
    logger.info(f"Wyoming bridge listening on {args.uri}")
    await server.run(partial(RivaEventHandler, backend=backend, info=info, catalog=catalog, settings=settings))


if __name__ == "__main__":
    asyncio.run(main())

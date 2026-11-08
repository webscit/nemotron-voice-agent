# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transcribe a recorded clip with the session's ASR (received Matrix voice messages).

The live pipeline streams the microphone to its ASR; this sends one clip through
the same Riva-compatible streaming RPC (``monitoring.jobs.asr_client``), outside
the pipeline. The gRPC channel opens on the first clip.
"""

from __future__ import annotations

import asyncio
import io
import wave

from monitoring.jobs.asr_client import AsrEndpoint, RivaTranscriber


class SessionTranscriber:
    """Async transcriber of 16-bit mono PCM clips, bound to one ASR endpoint."""

    def __init__(self, *, server: str, use_ssl: bool, language: str, model: str = "", function_id: str = ""):
        """Transcribe in ``language`` on the Riva-compatible ``server``."""
        self._endpoint = AsrEndpoint(
            name="session", server=server, use_ssl=use_ssl, language=language, model=model, function_id=function_id
        )
        self._language = language
        self._transcriber: RivaTranscriber | None = None

    async def __call__(self, pcm: bytes, sample_rate: int) -> str:
        """What the ASR hears in ``pcm``."""
        return await asyncio.to_thread(self._transcribe, pcm, sample_rate)

    def _transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if self._transcriber is None:
            self._transcriber = RivaTranscriber(self._endpoint)
        return self._transcriber.transcribe_wav(_wav(pcm, sample_rate), self._language)


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buffer.getvalue()

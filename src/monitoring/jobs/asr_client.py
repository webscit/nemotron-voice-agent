# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline transcription of recorded WAV clips through a Riva-compatible gRPC ASR.

Uses the streaming RPC (the one the live pipeline uses, so it works with Riva,
NIM and NeMo-Speech.cpp alike) and keeps final results only.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any

import grpc
import riva.client

from utils import is_nvcf, nvidia_api_key

_CHUNK_SECS = 0.1


@dataclass(frozen=True)
class AsrEndpoint:
    """One ASR variant to evaluate (``name`` becomes the annotation source)."""

    name: str
    server: str
    language: str | None = None
    model: str = ""
    function_id: str = ""
    service: str | None = None
    # Base language codes (e.g. ["en"]) this endpoint supports; empty = all.
    languages: tuple[str, ...] = ()

    def supports(self, language: str | None) -> bool:
        """Return whether this endpoint should transcribe a session in ``language``."""
        if not self.languages or not language:
            return True
        return language.split("-")[0].lower() in self.languages

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> AsrEndpoint:
        """Build from a ``dreamer.yaml`` entry."""
        return cls(
            name=str(raw["name"]),
            server=str(raw["server"]),
            language=raw.get("language"),
            model=str(raw.get("model", "") or ""),
            function_id=str(raw.get("function_id", "") or ""),
            service=raw.get("service"),
            languages=tuple(str(code).split("-")[0].lower() for code in raw.get("languages") or ()),
        )


class RivaTranscriber:
    """Blocking transcriber bound to one endpoint."""

    def __init__(self, endpoint: AsrEndpoint):
        """Open the gRPC channel to ``endpoint``."""
        self.endpoint = endpoint
        metadata = []
        if endpoint.function_id:
            metadata.append(["function-id", endpoint.function_id])
        if is_nvcf(endpoint.server):
            metadata.append(["authorization", f"Bearer {nvidia_api_key()}"])
        auth = riva.client.Auth(None, is_nvcf(endpoint.server), endpoint.server, metadata)
        self._channel = auth.channel
        self._service = riva.client.ASRService(auth)

    def wait_ready(self, timeout_secs: float = 300.0) -> None:
        """Block until the gRPC server accepts connections (freshly started containers)."""
        grpc.channel_ready_future(self._channel).result(timeout=timeout_secs)

    def transcribe_wav(self, wav_bytes: bytes, language: str) -> str:
        """Return the final transcript of a mono PCM16 WAV clip."""
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            sample_rate, channels = wf.getframerate(), wf.getnchannels()
            pcm = wf.readframes(wf.getnframes())
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self.endpoint.language or language,
                model=self.endpoint.model,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                sample_rate_hertz=sample_rate,
                audio_channel_count=channels,
            ),
            interim_results=False,
        )
        step = int(sample_rate * _CHUNK_SECS) * 2 * channels
        chunks = (pcm[i : i + step] for i in range(0, len(pcm), step))
        parts = []
        for response in self._service.streaming_response_generator(audio_chunks=chunks, streaming_config=config):
            for result in response.results:
                if result.is_final and result.alternatives:
                    parts.append(result.alternatives[0].transcript.strip())
        return " ".join(p for p in parts if p)

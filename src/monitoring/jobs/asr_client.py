# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline transcription of recorded WAV clips.

Two protocols are supported (``protocol`` in ``dreamer.yaml``):

- ``riva`` (default): Riva-compatible gRPC, using the streaming RPC the live
  pipeline uses (Riva, NIM and NeMo-Speech.cpp alike); final results only.
- ``openai``: OpenAI-compatible ``POST /v1/audio/transcriptions`` (vLLM serving
  Voxtral, Whisper, ...).
"""

from __future__ import annotations

import io
import time
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
    server: str = ""
    protocol: str = "riva"
    # OpenAI-compatible base URL (``protocol: openai``), e.g. http://host:8000/v1
    base_url: str = ""
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
            server=str(raw.get("server", "") or ""),
            protocol=str(raw.get("protocol", "riva")).lower(),
            base_url=str(raw.get("base_url", "") or ""),
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


class OpenAITranscriber:
    """Blocking transcriber for OpenAI-compatible ``/audio/transcriptions`` servers."""

    def __init__(self, endpoint: AsrEndpoint):
        """Create a client for ``endpoint.base_url`` (API key from ``OPENAI_API_KEY`` or a dummy)."""
        import os

        from openai import OpenAI

        self.endpoint = endpoint
        self._client = OpenAI(base_url=endpoint.base_url, api_key=os.getenv("OPENAI_API_KEY") or "not-needed")
        self._model = endpoint.model

    def wait_ready(self, timeout_secs: float = 900.0) -> None:
        """Poll ``/models`` until the server answers (vLLM loads the model first)."""
        deadline = time.monotonic() + timeout_secs
        while True:
            try:
                models = [m.id for m in self._client.models.list()]
                self._model = self._model or models[0]
                return
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(5.0)

    def transcribe_wav(self, wav_bytes: bytes, language: str) -> str:
        """Return the transcript of a WAV clip (``language`` sent as an ISO-639-1 code)."""
        code = (self.endpoint.language or language or "").split("-")[0].lower() or None
        kwargs = {"language": code} if code else {}
        result = self._client.audio.transcriptions.create(
            model=self._model, file=("turn.wav", wav_bytes, "audio/wav"), temperature=0.0, **kwargs
        )
        return (result.text or "").strip()


def make_transcriber(endpoint: AsrEndpoint) -> RivaTranscriber | OpenAITranscriber:
    """Return the transcriber matching ``endpoint.protocol``."""
    if endpoint.protocol == "openai":
        return OpenAITranscriber(endpoint)
    if endpoint.protocol == "riva":
        return RivaTranscriber(endpoint)
    raise ValueError(f"Unknown ASR protocol {endpoint.protocol!r} for {endpoint.name}")

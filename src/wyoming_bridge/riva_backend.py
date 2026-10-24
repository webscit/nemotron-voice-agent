# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blocking Riva gRPC calls used by the Wyoming bridge."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import grpc
import riva.client
from riva.client.proto import riva_asr_pb2, riva_tts_pb2

from examples.shared.prewarm import _parse_asr_config, _parse_tts_config

ASR_SAMPLE_RATE = 16000
# Home Assistant runs its own voice activity detection and ends the request with
# audio-stop, so Riva only needs to keep one command in one segment. With the
# server's short window (320 ms), words right after a mid-command pause can be lost.
DEFAULT_ASR_STOP_HISTORY_MS = 2000


@dataclass(frozen=True)
class SpeechCatalog:
    """What the Riva server reports it can recognize and synthesize."""

    asr_models: tuple[str, ...] = ()
    asr_languages: tuple[str, ...] = ()
    tts_languages: tuple[str, ...] = ()
    # Voice name -> languages it speaks, in server order.
    tts_voices: dict[str, tuple[str, ...]] = field(default_factory=dict)


def resolve_language(requested: str | None, available: Iterable[str], default: str) -> str:
    """Map a Wyoming language (``fr``, ``fr-FR``, or none) to a locale the server lists.

    Prefers an exact match, then ``default`` when it shares the base language, then the
    first listed locale with that base language. Falls back to ``default``.
    """
    if not requested:
        return default
    wanted = requested.replace("_", "-").lower()
    available = list(available)
    for code in available:
        if code.lower() == wanted:
            return code
    base = wanted.split("-", 1)[0]
    if default.split("-", 1)[0].lower() == base:
        return default
    for code in available:
        if code.split("-", 1)[0].lower() == base:
            return code
    return default


class RivaBackend:
    """Recognizes and synthesizes speech on one Riva gRPC endpoint (plaintext, on-prem)."""

    def __init__(self, server: str, asr_stop_history_ms: int = DEFAULT_ASR_STOP_HISTORY_MS):
        """Open the gRPC channel to ``server`` (``host:port``).

        ``asr_stop_history_ms`` is the trailing silence (a multiple of 80 ms) before Riva
        ends a segment; ``-1`` keeps the server setting.
        """
        self.server = server
        self._asr_stop_history_ms = asr_stop_history_ms
        auth = riva.client.Auth(None, False, server)
        self._channel = auth.channel
        self._asr = riva.client.ASRService(auth)
        self._tts = riva.client.SpeechSynthesisService(auth)

    def wait_ready(self, timeout_secs: float) -> None:
        """Block until the server accepts connections; raises ``grpc.FutureTimeoutError``."""
        grpc.channel_ready_future(self._channel).result(timeout=timeout_secs)

    def describe(self) -> SpeechCatalog:
        """Read the ASR and TTS model configs from the server."""
        asr_raw = self._asr.stub.GetRivaSpeechRecognitionConfig(riva_asr_pb2.RivaSpeechRecognitionConfigRequest())
        tts_raw = self._tts.stub.GetRivaSynthesisConfig(riva_tts_pb2.RivaSynthesisConfigRequest())
        tts = _parse_tts_config(tts_raw, "")
        voices: dict[str, list[str]] = {}
        for voice in tts["voices"]:
            voices.setdefault(voice["id"], []).append(voice["language"])
        return SpeechCatalog(
            asr_models=tuple(mc.model_name for mc in asr_raw.model_config),
            asr_languages=tuple(_parse_asr_config(asr_raw)["languages"]),
            tts_languages=tuple(tts["languages"]),
            tts_voices={name: tuple(langs) for name, langs in voices.items()},
        )

    def transcribe(self, chunks: Iterable[bytes], language: str) -> str:
        """Stream 16 kHz mono PCM16 ``chunks`` and return the joined final transcripts."""
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=language,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                sample_rate_hertz=ASR_SAMPLE_RATE,
                audio_channel_count=1,
            ),
            interim_results=False,
        )
        stop = self._asr_stop_history_ms
        riva.client.add_endpoint_parameters_to_config(config, -1, -1.0, stop, stop, -1.0, -1.0)
        parts = []
        for response in self._asr.streaming_response_generator(audio_chunks=chunks, streaming_config=config):
            for result in response.results:
                if result.is_final and result.alternatives:
                    parts.append(result.alternatives[0].transcript.strip())
        return " ".join(p for p in parts if p)

    def synthesize(self, text: str, voice: str | None, language: str, sample_rate: int) -> Iterator[bytes]:
        """Yield mono PCM16 audio for ``text`` as the server produces it."""
        for response in self._tts.synthesize_online(
            text,
            voice_name=voice or None,
            language_code=language,
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=sample_rate,
        ):
            if response.audio:
                yield response.audio

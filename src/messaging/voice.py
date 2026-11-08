# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice-message audio: decode what contacts send, encode what the agent records.

Audio inside the pipeline is 16-bit mono PCM. Matrix voice messages are Ogg/Opus
(``audio/ogg``), the format Element records; decoding accepts anything FFmpeg
reads, since other clients send AAC or MP3.
"""

from __future__ import annotations

import io
import math
from array import array

import av

# Opus is encoded at 48 kHz whatever the input rate (the rate every decoder supports).
OPUS_SAMPLE_RATE = 48_000
OPUS_BIT_RATE = 32_000
OGG_MIMETYPE = "audio/ogg"
# Points of the waveform Element draws for a voice message (values 0..1024, MSC3246).
WAVEFORM_POINTS = 64
WAVEFORM_MAX = 1024
_BYTES_PER_SAMPLE = 2


class VoiceCodecError(ValueError):
    """The audio could not be decoded or encoded."""


def pcm_duration_ms(pcm: bytes, sample_rate: int) -> int:
    """Duration of 16-bit mono PCM, in milliseconds."""
    return len(pcm) * 1000 // (_BYTES_PER_SAMPLE * sample_rate)


def decode_audio(data: bytes, sample_rate: int, *, max_secs: float | None = None) -> bytes:
    """Decode an audio file to 16-bit mono PCM at ``sample_rate``, cut after ``max_secs``."""
    limit = int(max_secs * sample_rate) * _BYTES_PER_SAMPLE if max_secs else None
    out = bytearray()
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.audio:
                raise VoiceCodecError("no audio stream")
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    out += bytes(resampled.planes[0])[: resampled.samples * _BYTES_PER_SAMPLE]
                if limit is not None and len(out) >= limit:
                    break
            else:
                for resampled in resampler.resample(None):
                    out += bytes(resampled.planes[0])[: resampled.samples * _BYTES_PER_SAMPLE]
    except av.FFmpegError as exc:
        raise VoiceCodecError(f"cannot decode the audio: {exc}") from exc
    return bytes(out[:limit] if limit is not None else out)


def encode_voice(pcm: bytes, sample_rate: int) -> bytes:
    """Encode 16-bit mono PCM as an Ogg/Opus voice message."""
    if not pcm:
        raise VoiceCodecError("no audio to encode")
    buffer = io.BytesIO()
    try:
        with av.open(buffer, mode="w", format="ogg") as container:
            stream = container.add_stream("libopus", rate=OPUS_SAMPLE_RATE, layout="mono")
            stream.bit_rate = OPUS_BIT_RATE
            resampler = av.AudioResampler(format="s16", layout="mono", rate=OPUS_SAMPLE_RATE)
            frame = av.AudioFrame(format="s16", layout="mono", samples=len(pcm) // _BYTES_PER_SAMPLE)
            frame.planes[0].update(pcm[: frame.samples * _BYTES_PER_SAMPLE])
            frame.sample_rate = sample_rate
            frame.pts = 0
            for resampled in [*resampler.resample(frame), *resampler.resample(None)]:
                for packet in stream.encode(resampled):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
    except av.FFmpegError as exc:
        raise VoiceCodecError(f"cannot encode the audio: {exc}") from exc
    return buffer.getvalue()


def waveform(pcm: bytes, points: int = WAVEFORM_POINTS) -> list[int]:
    """Loudness of ``points`` equal slices of 16-bit PCM, scaled to 0..1024 (the loudest slice)."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % _BYTES_PER_SAMPLE])
    if not samples:
        return []
    points = min(points, len(samples))
    step = len(samples) / points
    levels = []
    for i in range(points):
        chunk = samples[int(i * step) : int((i + 1) * step)] or samples[int(i * step) : int(i * step) + 1]
        levels.append(math.sqrt(sum(s * s for s in chunk) / len(chunk)))
    peak = max(levels)
    if peak <= 0:
        return [0] * points
    return [round(level / peak * WAVEFORM_MAX) for level in levels]

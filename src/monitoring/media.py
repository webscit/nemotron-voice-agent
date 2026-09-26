# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers turning images/audio seen in a session into artifacts + ``media`` rows.

Everything here runs in the writer thread (hashing, base64 decoding, JPEG and
WAV encoding), never on the pipeline event loop.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import wave
from typing import Any

from PIL import Image

_MIME_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "audio/wav": "wav",
    "video/mp4": "mp4",
}


def sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def extension_for(mime: str) -> str:
    """Return the file extension used to store a payload of type ``mime``."""
    return _MIME_EXT.get(mime, mime.split("/")[-1] or "bin")


def encode_raw_image(
    image: bytes, size: tuple[int, int], fmt: str | None, *, max_width: int | None = None, quality: int = 85
) -> tuple[bytes, str, int, int]:
    """Return ``(encoded, mime, width, height)`` for a pipecat image frame payload."""
    fmt = fmt or "RGB"
    if fmt.startswith("image/") and max_width is None:
        with Image.open(io.BytesIO(image)) as img:
            return image, fmt, img.width, img.height
    if fmt.startswith("image/"):
        img = Image.open(io.BytesIO(image))
    else:
        img = Image.frombytes(fmt, size, image)
    if max_width and img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue(), "image/jpeg", img.width, img.height


def image_dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Return ``(width, height)`` of an encoded image, or ``(None, None)`` if undecodable."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.width, img.height
    except Exception:
        return None, None


def pcm16_to_wav(audio: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw PCM16 samples in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    return buffer.getvalue()


def parse_data_url(value: str) -> tuple[str, bytes] | None:
    """Decode ``data:<mime>;base64,<payload>``; return None for anything else."""
    if not value.startswith("data:") or ";base64," not in value[:128]:
        return None
    header, _, payload = value.partition(",")
    mime = header[5:].split(";")[0] or "application/octet-stream"
    try:
        return mime, base64.b64decode(payload, validate=False)
    except ValueError:
        return None


@dataclasses.dataclass
class ExtractedMedia:
    """Inline media payload found in an LLM input or tool result."""

    sha256: str
    mime: str
    data: bytes


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of context/tool payloads into JSON-safe values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, bytes | bytearray):
        return {"type": "bytes", "length": len(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    return str(value)


def extract_inline_media(value: Any) -> tuple[Any, list[ExtractedMedia]]:
    """Replace inline ``data:`` media (image/audio/video) with ``media_ref`` stubs.

    Walks any JSON-like structure (OpenAI ``image_url`` parts, tool results, ...)
    and returns the stripped structure plus the extracted payloads, so stored LLM
    inputs stay small while remaining replayable through the ``media`` table.
    Anthropic-style ``{"type": "base64", "media_type", "data"}`` sources are also
    handled.
    """
    found: list[ExtractedMedia] = []

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            decoded = parse_data_url(node) if node.startswith("data:") else None
            if decoded and decoded[0].split("/")[0] in ("image", "audio", "video"):
                mime, data = decoded
                digest = sha256(data)
                found.append(ExtractedMedia(digest, mime, data))
                return {"type": "media_ref", "sha256": digest, "mime": mime}
            return node
        if isinstance(node, dict):
            if node.get("type") == "base64" and isinstance(node.get("data"), str) and "media_type" in node:
                try:
                    data = base64.b64decode(node["data"])
                except ValueError:
                    data = None
                if data is not None:
                    digest = sha256(data)
                    found.append(ExtractedMedia(digest, str(node["media_type"]), data))
                    return {"type": "media_ref", "sha256": digest, "mime": node["media_type"]}
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(to_jsonable(value)), found

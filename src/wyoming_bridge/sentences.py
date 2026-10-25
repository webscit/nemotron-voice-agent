# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental sentence splitting for streaming text-to-speech."""

from __future__ import annotations

import re

# A sentence ends at terminal punctuation (optionally followed by closing quotes or
# brackets, with the French space before ``»``) and then whitespace, or at a line
# break. Decimals such as ``3.5`` and times such as ``15:30`` have no whitespace after
# the mark, so they stay intact.
_BOUNDARY = re.compile(r"[.!?…]+(?:\s?[\"'»”)\]])*\s+|\n+")


class SentenceSplitter:
    """Buffers streamed text chunks and releases complete sentences."""

    def __init__(self) -> None:
        """Start with an empty buffer."""
        self._buffer = ""

    def add(self, text: str) -> list[str]:
        """Append ``text`` and return the sentences it completed."""
        self._buffer += text
        sentences = []
        start = 0
        for match in _BOUNDARY.finditer(self._buffer):
            sentence = self._buffer[start : match.end()].strip()
            if sentence:
                sentences.append(sentence)
            start = match.end()
        self._buffer = self._buffer[start:]
        return sentences

    def finish(self) -> list[str]:
        """Return the unterminated remainder, if any, and reset the buffer."""
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Word / character error rate with light, language-neutral normalization."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s']|_", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, NFKC-normalize, and strip punctuation (apostrophes kept)."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(_PUNCT.sub(" ", text).split())


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str) -> dict[str, float | int]:
    """Return WER/CER and the raw counts needed to aggregate them correctly."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    ref_words, hyp_words = ref.split(), hyp.split()
    ref_chars, hyp_chars = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    word_errors = _edit_distance(ref_words, hyp_words)
    char_errors = _edit_distance(ref_chars, hyp_chars)
    return {
        "wer": word_errors / len(ref_words) if ref_words else float(bool(hyp_words)),
        "cer": char_errors / len(ref_chars) if ref_chars else float(bool(hyp_chars)),
        "word_errors": word_errors,
        "ref_words": len(ref_words),
        "char_errors": char_errors,
        "ref_chars": len(ref_chars),
    }

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Language-aware TTS text normalization built on Pipecat's text transforms.

Pipecat's :class:`~pipecat.utils.text.transforms.VoiceFormatter` only speaks English. This
formatter picks a transform chain from the TTS session language on every call:

* ``en`` — Pipecat's ``VoiceFormatter`` defaults, without acronym letter-spacing.
* ``fr`` — Pipecat's language-neutral transforms plus the French counterparts in
  :mod:`examples.shared.tts_text_transforms_fr`.
* any other language — text passes through unchanged.

Register it with ``tts.add_text_transformer(...)``: transforms only change the text sent to the
TTS engine; the transcript and LLM context keep the original text.
"""

from collections.abc import Awaitable, Callable

from pipecat.frames.frames import AggregationType
from pipecat.utils.text.transforms import VoiceFormatter, strip_markdown

from examples.shared import tts_text_transforms_fr as fr

Transform = Callable[[str, str | AggregationType], Awaitable[str]]

# Same order as Pipecat's VoiceFormatter: structural cleanup first, then language expansions.
# Times run before currency/units so "15h30" is not mistaken for another pattern. Pipecat's
# acronym letter-spacing is left out in both languages: it turns brand names such as "NVIDIA"
# into "N V I D I A".
_FRENCH_TRANSFORMS: tuple[Transform, ...] = (
    strip_markdown,
    fr.email_to_speech,
    fr.expand_phone_numbers,
    fr.normalize_dates,
    fr.expand_times,
    fr.expand_currency,
    fr.expand_percentages,
    fr.expand_units,
    fr.join_thousands,
)


class LanguageAwareVoiceFormatter:
    """Apply the English or French voice-formatting chain for the current TTS language."""

    def __init__(self, language_getter: Callable[[], object]):
        """Initialize the formatter.

        Args:
            language_getter: Returns the current TTS language (for example ``"fr-FR"`` or a
                Pipecat ``Language``). Read on every call so mid-session voice/language
                switches are honored.
        """
        self._language_getter = language_getter
        self._chains: dict[str, tuple[Transform, ...]] = {
            "en": (VoiceFormatter(normalize_acronyms=False),),
            "fr": _FRENCH_TRANSFORMS,
        }

    @staticmethod
    def base_language(language: object) -> str:
        """Return the lowercase base language code (``"fr-FR"`` -> ``"fr"``)."""
        code = str(getattr(language, "value", language) or "")
        return code.replace("_", "-").split("-", 1)[0].strip().lower()

    async def __call__(self, text: str, aggregation_type: str | AggregationType) -> str:
        """Normalize ``text`` for the current TTS language."""
        for transform in self._chains.get(self.base_language(self._language_getter()), ()):
            text = await transform(text, aggregation_type)
        return text

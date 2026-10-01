# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

import asyncio
import unittest

from pipecat.transcriptions.language import Language
from pipecat.utils.text.transforms import VoiceFormatter

from examples.shared import tts_text_transforms_fr as fr
from examples.shared.tts_text_normalizer import LanguageAwareVoiceFormatter


def run(transform, text: str) -> str:
    return asyncio.run(transform(text, "sentence"))


class FrenchTransformTests(unittest.TestCase):
    def assertTransforms(self, transform, cases: dict[str, str]) -> None:
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(run(transform, text), expected)

    def test_expand_currency(self) -> None:
        self.assertTransforms(
            fr.expand_currency,
            {
                "Cela coûte 12,50 €.": "Cela coûte douze euros cinquante.",
                "Cela coûte 12€.": "Cela coûte douze euros.",
                "Cela coûte 1 €.": "Cela coûte un euro.",
                "Cela coûte €3.": "Cela coûte trois euros.",
                "Budget : 3 000 €.": "Budget : trois mille euros.",
                "Il gagne 1 000 000 €.": "Il gagne un million d’euros.",
                "Soit $3,20.": "Soit trois dollars vingt cents.",
                "Soit 5 £.": "Soit cinq livres.",
                "Pas de prix ici.": "Pas de prix ici.",
            },
        )

    def test_expand_percentages(self) -> None:
        self.assertTransforms(
            fr.expand_percentages,
            {
                "25 % de plus": "vingt-cinq pour cent de plus",
                "12,5% de moins": "douze virgule cinq pour cent de moins",
            },
        )

    def test_expand_units(self) -> None:
        self.assertTransforms(
            fr.expand_units,
            {
                "Il reste 3,5 km.": "Il reste trois virgule cinq kilomètres.",
                "Roulez à 130 km/h.": "Roulez à cent trente kilomètres heure.",
                "Il fait 21 °C.": "Il fait vingt et un degrés Celsius.",
                "Un fichier de 5 Go et 1 kg.": "Un fichier de cinq gigaoctets et un kilo.",
                "Attendez 5 minutes.": "Attendez 5 minutes.",
                "Version 3.5 m": "Version 3.5 m",
            },
        )

    def test_normalize_dates(self) -> None:
        self.assertTransforms(
            fr.normalize_dates,
            {
                "le 03/03/2024": "le trois mars deux mille vingt-quatre",
                "le 2024-12-01": "le premier décembre deux mille vingt-quatre",
                "le 31/13/2024": "le 31/13/2024",
            },
        )

    def test_expand_times(self) -> None:
        self.assertTransforms(
            fr.expand_times,
            {
                "à 15h30": "à quinze heures trente",
                "à 15 h 30": "à quinze heures trente",
                "à 9h.": "à neuf heures.",
                "à 1h05": "à une heure cinq",
                "à 21h": "à vingt et une heures",
                "à 00:30": "à zéro heure trente",
                "15 habitants": "15 habitants",
            },
        )

    def test_expand_phone_numbers(self) -> None:
        self.assertTransforms(
            fr.expand_phone_numbers,
            {
                "le 06 12 34 56 78": "le zéro six douze trente-quatre cinquante-six soixante-dix-huit",
                "le +33 1 02 03 04 05": "le zéro un zéro deux zéro trois zéro quatre zéro cinq",
                "le 0612345678": "le zéro six douze trente-quatre cinquante-six soixante-dix-huit",
            },
        )

    def test_email_to_speech(self) -> None:
        self.assertEqual(
            run(fr.email_to_speech, "Écrivez à jean.dupont@example.fr"),
            "Écrivez à jean point dupont arobase example point fr",
        )

    def test_join_thousands(self) -> None:
        self.assertTransforms(
            fr.join_thousands,
            {
                "Il y a 1 234 personnes.": "Il y a 1234 personnes.",
                "Il y a 12 345 personnes.": "Il y a 12345 personnes.",
                "Il y a 12 personnes.": "Il y a 12 personnes.",
            },
        )


class LanguageAwareVoiceFormatterTests(unittest.TestCase):
    def test_french_chain(self) -> None:
        formatter = LanguageAwareVoiceFormatter(lambda: "fr-FR")
        self.assertEqual(
            run(formatter, "**Cela** coûte 12,50 € à 15h30 chez NVIDIA."),
            "Cela coûte douze euros cinquante à quinze heures trente chez NVIDIA.",
        )

    def test_english_uses_pipecat_voice_formatter_without_acronyms(self) -> None:
        text = "NVIDIA charges $12.50, 50% off, 5km on 2023-05-10."
        expected = run(VoiceFormatter(normalize_acronyms=False), text)
        self.assertEqual(run(LanguageAwareVoiceFormatter(lambda: "en-US"), text), expected)
        self.assertIn("NVIDIA", expected)
        self.assertIn("twelve dollars", expected)

    def test_accepts_pipecat_language_enum(self) -> None:
        formatter = LanguageAwareVoiceFormatter(lambda: Language.FR_FR)
        self.assertEqual(run(formatter, "25 %"), "vingt-cinq pour cent")

    def test_unsupported_language_passes_through(self) -> None:
        for language in ("de-DE", "", None):
            with self.subTest(language=language):
                formatter = LanguageAwareVoiceFormatter(lambda language=language: language)
                self.assertEqual(run(formatter, "Es kostet 12,50 €."), "Es kostet 12,50 €.")

    def test_follows_language_switch(self) -> None:
        language = ["en-US"]
        formatter = LanguageAwareVoiceFormatter(lambda: language[0])
        self.assertIn("percent", run(formatter, "25%"))
        language[0] = "fr-FR"
        self.assertEqual(run(formatter, "25%"), "vingt-cinq pour cent")


if __name__ == "__main__":
    unittest.main()

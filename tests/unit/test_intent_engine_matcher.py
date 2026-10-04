# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from examples.shared.intent_engine.config import DEFAULT_ALLOWED_INTENTS, PACKAGED_SENTENCES_DIR, IntentEngineConfig
from examples.shared.intent_engine.matcher import IntentMatcher, build_slot_lists, normalize_text
from examples.shared.intent_engine.targets import ClientToolMap

WAKE_WORDS = ("Reachy", "Richie", "Retchi", "Recchi", "Itty")
ENTITIES = [
    {"id": "light.lampe_du_salon", "name": "Lampe du salon", "domain": "light", "device_class": None},
    {"id": "media_player.tele", "name": "Télé", "domain": "media_player", "device_class": "tv"},
    {"id": "cover.volets", "name": "Volets!", "domain": "cover", "device_class": "shutter"},
    {"id": "sensor.unnamed", "name": "", "domain": "sensor", "device_class": None},
]
SLOT_LISTS = build_slot_lists(ENTITIES, ["Salon", "Cuisine"], ["Étage"])
REACHY_TOOLS = {"volume_control", "move_head", "dance", "stop_dance", "play_emotion", "go_to_sleep", "sweep_look"}


def make_matcher(
    language="fr-FR", *, tools=REACHY_TOOLS, area="Salon", allowed=DEFAULT_ALLOWED_INTENTS, directory=None
):
    directory = directory or PACKAGED_SENTENCES_DIR
    tool_map = ClientToolMap.load(directory)

    def keep(name: str) -> bool:
        return tool_map.tool(name) in tools if name in tool_map else name in allowed

    return IntentMatcher(
        language,
        allowed_intents=allowed,
        sentences_dir=directory,
        wake_words=WAKE_WORDS,
        area=area,
        keep_custom_intent=keep,
    )


class NormalizeTextTests(unittest.TestCase):
    def test_strips_leading_greeting_and_wake_word(self) -> None:
        cases = {
            "Bonjour Reachy, allume la lampe": "allume la lampe",
            "Richie allume la lampe": "allume la lampe",
            "Merci Retchi,  tu peux aller dormir.": "tu peux aller dormir.",
            "bonjour, allume la lampe": "allume la lampe",
            "  Allume   la lampe ": "Allume la lampe",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize_text(text, "fr-FR", WAKE_WORDS), expected)

    def test_keeps_text_when_nothing_would_remain_or_no_prefix(self) -> None:
        self.assertEqual(normalize_text("Bonjour Reachy", "fr-FR", WAKE_WORDS), "Bonjour Reachy")
        self.assertEqual(normalize_text("Reachyallume", "fr-FR", WAKE_WORDS), "Reachyallume")
        self.assertEqual(normalize_text("allume Reachy", "fr-FR", WAKE_WORDS), "allume Reachy")
        self.assertEqual(normalize_text("[session_start]", "fr-FR", WAKE_WORDS), "[session_start]")
        self.assertEqual(normalize_text("", "fr-FR", WAKE_WORDS), "")

    def test_wake_words_apply_to_languages_without_greeting_list(self) -> None:
        self.assertEqual(normalize_text("Reachy, schalte das Licht ein", "de-DE", WAKE_WORDS), "schalte das Licht ein")


class SlotListTests(unittest.TestCase):
    def test_name_list_carries_domain_and_device_class_context(self) -> None:
        names = {value.value_out: value.context for value in SLOT_LISTS["name"].values}
        self.assertEqual(names["Lampe du salon"], {"domain": "light"})
        self.assertEqual(names["Télé"], {"domain": "media_player", "device_class": "tv"})
        # Punctuation is stripped from the spoken form, not from the value sent to Home Assistant.
        self.assertIn("Volets!", names)
        self.assertEqual(len(names), 3)  # the unnamed entity is skipped

    def test_blocked_domains_never_enter_the_name_list(self) -> None:
        entities = [
            *ENTITIES,
            {"id": "lock.front_door", "name": "Front Door", "domain": "lock"},
            {"id": "binary_sensor.front_door", "name": "front door", "domain": "binary_sensor"},
            {"id": "alarm_control_panel.home", "name": "Alarme", "domain": "alarm_control_panel"},
        ]
        lists = build_slot_lists(entities, ["Salon"], [], ("lock", "alarm_control_panel"))
        names = [value.value_out for value in lists["name"].values]
        # The sensor sharing the lock's name goes too: Home Assistant would match both by name.
        self.assertEqual(names, ["Lampe du salon", "Télé", "Volets!"])
        matcher = make_matcher(tools=set())
        for text in ("Verrouille Front Door", "Déverrouille Front Door", "Allume Front Door", "Éteins Alarme"):
            with self.subTest(text=text):
                self.assertIsNone(matcher.match(text, lists))
        unblocked = build_slot_lists(entities, ["Salon"], [])
        self.assertEqual(matcher.match("Déverrouille Front Door", unblocked).name, "HassTurnOff")

    def test_area_and_floor_lists(self) -> None:
        self.assertEqual([v.value_out for v in SLOT_LISTS["area"].values], ["Salon", "Cuisine"])
        self.assertEqual([v.value_out for v in SLOT_LISTS["floor"].values], ["Étage"])


class StockMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matcher = make_matcher(tools=set())

    def match(self, text: str):
        return self.matcher.match(text, SLOT_LISTS)

    def test_name_match_and_polite_form(self) -> None:
        for text in ("Allume la lampe du salon.", "Est-ce que tu peux allumer la lampe du salon ?"):
            with self.subTest(text=text):
                match = self.match(text)
                self.assertEqual(
                    (match.name, match.slots, match.custom), ("HassTurnOn", {"name": "Lampe du salon"}, False)
                )

    def test_wake_word_prefix_is_stripped_before_matching(self) -> None:
        self.assertEqual(self.match("Bonjour Recchi, allume la lampe du salon").name, "HassTurnOn")

    def test_area_context_fills_area_slot(self) -> None:
        match = self.match("Allume les lumières")
        self.assertEqual((match.name, match.slots), ("HassTurnOn", {"domain": "light", "area": "Salon"}))
        self.assertIsNone(make_matcher(tools=set(), area="").match("Allume les lumières", SLOT_LISTS))

    def test_spoken_numbers_and_response_key(self) -> None:
        match = self.match("mets le volume de la télé à quatre-vingts pour cent")
        self.assertEqual((match.name, match.slots["volume_level"]), ("HassSetVolume", 80))
        match = self.match("Quel est l'état de la lampe du salon ?")
        self.assertEqual((match.name, match.response_key), ("HassGetState", "one"))

    def test_wildcard_and_timer_intents_are_pruned(self) -> None:
        # With the area context these sentences match HassMediaSearchAndPlay / HassStartTimer upstream.
        for text in ("Mets le volume à soixante-dix", "Mets un minuteur de cinq minutes", "Mets du jazz"):
            with self.subTest(text=text):
                self.assertIsNone(self.match(text))

    def test_allow_list_override_restricts_matches(self) -> None:
        matcher = make_matcher(tools=set(), allowed=("HassGetCurrentTime",))
        self.assertIsNone(matcher.match("Allume la lampe du salon", SLOT_LISTS))
        self.assertEqual(matcher.match("Quelle heure est-il ?", SLOT_LISTS).name, "HassGetCurrentTime")

    def test_stock_sentences_need_synced_slot_lists(self) -> None:
        self.assertIsNone(self.matcher.match("Quelle heure est-il ?", None))

    def test_compound_and_chat_sentences_do_not_match(self) -> None:
        for text in (
            "Allume la lampe du salon et ferme les volets",
            "Que connais-tu sur la ville de Rennes en France?",
            "Merci",
            "[session_start]",
        ):
            with self.subTest(text=text):
                self.assertIsNone(self.match(text))

    def test_templates_come_from_the_language_package(self) -> None:
        self.assertEqual(self.matcher.response_template("HassTurnOn", "cover"), "Ouverture en cours")
        self.assertEqual(self.matcher.error_template(), "Une erreur est intervenue pendant le traitement")
        self.assertIsNone(self.matcher.response_template("HassTurnOn", "nope"))


class LanguageTests(unittest.TestCase):
    def test_region_code_maps_to_base_language(self) -> None:
        match = make_matcher("en-US", tools=set(), area="").match(
            "turn on the lamp", build_slot_lists([{"name": "Lamp", "domain": "light"}], [], [])
        )
        self.assertEqual(match.name, "HassTurnOn")

    def test_without_stock_sentences_only_custom_intents_match(self) -> None:
        tool_map = ClientToolMap.load(PACKAGED_SENTENCES_DIR)
        matcher = IntentMatcher(
            "fr-FR",
            allowed_intents=DEFAULT_ALLOWED_INTENTS,
            sentences_dir=PACKAGED_SENTENCES_DIR,
            keep_custom_intent=lambda name: name in tool_map,
            load_stock=False,
        )
        self.assertTrue(matcher.has_data)
        self.assertIsNone(matcher.match("Allume la lampe du salon", SLOT_LISTS))
        self.assertEqual(matcher.match("Va dormir", None).name, "ReachyGoToSleep")
        # The localized error sentence is still available for client-tool failures.
        self.assertEqual(matcher.error_template(), "Une erreur est intervenue pendant le traitement")

    def test_language_without_data_never_matches(self) -> None:
        matcher = make_matcher("xx-YY")
        self.assertFalse(matcher.has_data)
        self.assertIsNone(matcher.match("allume la lampe du salon", SLOT_LISTS))


class CustomSentenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matcher = make_matcher()
        cls.tool_map = ClientToolMap.load(PACKAGED_SENTENCES_DIR)

    def call(self, text: str, slot_lists=SLOT_LISTS):
        match = self.matcher.match(text, slot_lists)
        if match is None or not match.custom:
            return match
        if self.tool_map.defers_to_llm(match.name):
            return "llm"
        return self.tool_map.tool(match.name), self.tool_map.arguments(match.name, match.slots)

    def test_reachy_sentences_from_recordings(self) -> None:
        cases = {
            "Va dormir.": ("go_to_sleep", {}),
            "Merci Richie, tu peux aller dormir.": ("go_to_sleep", {}),
            "Est-ce que tu peux regarder à droite?": ("move_head", {"direction": "right"}),
            "Est-ce que tu peux regarder sur ta gauche, s'il te plaît?": ("move_head", {"direction": "left"}),
            "Lève la tête": ("move_head", {"direction": "up"}),
            "Regarde en bas": ("move_head", {"direction": "down"}),
            "Regarde devant toi": ("move_head", {"direction": "front"}),
            "Remets la tête droite": ("move_head", {"direction": "front"}),
            # Without a move the client picks a random dance; "repeat" is optional too.
            "Est-ce que tu peux danser?": ("dance", {}),
            "Peux-tu faire une petite danse?": ("dance", {}),
            "Est-ce que tu peux faire une autre danse?": ("dance", {}),
            "Danse trois fois": ("dance", {"repeat": 3}),
            "Arrête de danser": ("stop_dance", {"dummy": True}),
            "Peux-tu faire une émotion comme être surpris?": ("play_emotion", {"emotion": "surprised"}),
            "Peux-tu montrer une  Émotion surpris.": ("play_emotion", {"emotion": "surprised"}),
            "Montre-moi une émotion": ("play_emotion", {}),
            "Sois triste": ("play_emotion", {"emotion": "sad"}),
            "Montre-moi la colère": ("play_emotion", {"emotion": "angry"}),
            "Fais une émotion de joie": ("play_emotion", {"emotion": "happy"}),
            "Montre que tu es fatigué": ("play_emotion", {"emotion": "tired"}),
            "Regarde autour de toi": ("sweep_look", {}),
            "Mettons volume à soixante-dix": ("volume_control", {"device": "speaker", "level": 70}),
            "Mets le volume à 70 %": ("volume_control", {"device": "speaker", "level": 70}),
            "Mets ton volume au maximum": ("volume_control", {"device": "speaker", "level": 100}),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.call(text), expected)

    def test_mapped_values_exist_in_the_reachy_tool_schemas(self) -> None:
        # Enums copied from reachy_mini_conversation_app/tools (move_head.py, play_emotion.py).
        directions = {"left", "right", "up", "down", "front"}
        emotions = {
            *("random", "happy", "excited", "loving", "grateful", "success", "thinking", "attentive", "confused"),
            *("uncertain", "sad", "downcast", "lonely", "angry", "irritated", "displeased", "disgusted", "scared"),
            *("anxious", "surprised", "amazed", "calming", "relief", "impatient", "embarrassed", "bored", "tired"),
            *("sleepy", "yes", "yes_understanding", "no", "no_sad", "no_excited", "no_firm", "welcoming"),
            *("greeting", "goodbye", "go_away", "helpful", "dance", "electric", "dying"),
        }
        content = yaml.safe_load((PACKAGED_SENTENCES_DIR / "fr" / "reachy_mini.yaml").read_text(encoding="utf-8"))
        self.assertLessEqual({value["out"] for value in content["lists"]["direction"]["values"]}, directions)
        self.assertLessEqual({value["out"] for value in content["lists"]["emotion"]["values"]}, emotions)
        fixed = {
            block["slots"]["direction"] for block in content["intents"]["ReachyMoveHead"]["data"] if "slots" in block
        }
        self.assertLessEqual(fixed, directions)

    def test_custom_sentences_work_without_home_assistant_names(self) -> None:
        self.assertEqual(self.call("Va dormir.", None), ("go_to_sleep", {}))

    def test_relative_volume_reads_the_level_then_shifts_it(self) -> None:
        cases = {
            "Parle un peu plus fort.": 10,
            "Parle encore plus fort.": 20,
            "Monte le volume": 20,
            "moins fort": -20,
            "Baisse un peu le son": -10,
        }
        for text, step in cases.items():
            with self.subTest(text=text):
                match = self.matcher.match(text, SLOT_LISTS)
                self.assertEqual((match.name, match.slots), ("ReachyVolumeRelative", {"step": step}))
                # First call reads (no level); the second sets the shifted, clamped level.
                self.assertEqual(self.call(text), ("volume_control", {"device": "speaker"}))
                self.assertTrue(self.tool_map.has_follow_up(match.name))
        self.assertEqual(
            self.tool_map.follow_up_arguments(
                "ReachyVolumeRelative", {"step": 20}, {"device": "speaker", "volume": 90}
            ),
            {"device": "speaker", "level": 100},
        )
        self.assertEqual(
            self.tool_map.follow_up_arguments("ReachyVolumeRelative", {"step": -10}, {"volume": 45}),
            {"device": "speaker", "level": 35},
        )
        for bad in ({"volume": None}, {"error": "x"}, None, {"volume": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.tool_map.follow_up_arguments("ReachyVolumeRelative", {"step": 10}, bad)

    def test_custom_sentences_take_priority_over_stock(self) -> None:
        # Without the Reachy tool this is HassSetVolumeRelative in the agent's area.
        stock = make_matcher(tools=set()).match("Monte le volume", SLOT_LISTS)
        self.assertEqual((stock.name, stock.custom), ("HassSetVolumeRelative", False))
        self.assertTrue(self.matcher.match("Monte le volume", SLOT_LISTS).custom)

    def test_intent_is_inactive_when_its_tool_was_not_declared(self) -> None:
        matcher = make_matcher(tools={"move_head"})
        self.assertEqual(matcher.custom_intent_names, ["ReachyMoveHead"])
        self.assertIsNone(matcher.match("Va dormir.", SLOT_LISTS))

    def test_home_assistant_names_still_match_next_to_custom_sentences(self) -> None:
        match = self.matcher.match("Allume la lampe du salon", SLOT_LISTS)
        self.assertEqual((match.name, match.custom), ("HassTurnOn", False))

    def test_custom_directory_can_extend_home_assistant_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            language_dir = Path(tmp) / "fr"
            language_dir.mkdir()
            (language_dir / "home.yaml").write_text(
                "language: fr\n"
                "intents:\n"
                "  HassTurnOn:\n"
                "    data:\n"
                "      - sentences: ['que la lumière soit']\n"
                "        slots: {domain: light}\n"
                "        response: default\n"
                "  HassStartTimer:\n"
                "    data:\n"
                "      - sentences: ['minuteur oeuf']\n"
                "  Broken: 3\n",
                encoding="utf-8",
            )
            (language_dir / "bad.yaml").write_text("intents: [unclosed", encoding="utf-8")
            matcher = make_matcher(directory=Path(tmp))
            match = matcher.match("Que la lumière soit !", SLOT_LISTS)
            self.assertEqual((match.name, match.slots, match.custom), ("HassTurnOn", {"domain": "light"}, True))
            self.assertEqual(matcher.response_template(match), "Allumé")
            # Not on the allow-list and not mapped to a client tool: dropped at load time.
            self.assertEqual(matcher.custom_intent_names, ["HassTurnOn"])


class ClientToolMapTests(unittest.TestCase):
    def test_arguments_copy_slots_apply_defaults_and_keep_literals(self) -> None:
        tool_map = ClientToolMap(
            {
                "A": {
                    "tool": "t",
                    "arguments": {"fixed": 1, "copied": {"slot": "x"}, "fallback": {"slot": "y", "default": 2}},
                },
                "B": {"tool": "t", "arguments": {"missing": {"slot": "nope"}}},
                "C": {"tool": "t", "defer_to_llm": True},
                "bad": "not a mapping",
            }
        )
        self.assertEqual(tool_map.arguments("A", {"x": "left"}), {"fixed": 1, "copied": "left", "fallback": 2})
        self.assertEqual(tool_map.arguments("B", {}), {})
        self.assertTrue(tool_map.defers_to_llm("C"))
        self.assertNotIn("bad", tool_map)

    def test_missing_mapping_file_is_an_empty_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotIn("ReachyDance", ClientToolMap.load(Path(tmp)))


class ConfigTests(unittest.TestCase):
    def env(self, **values: str):
        names = [name for name in os.environ if name.startswith(("INTENT_ENGINE_", "HOME_ASSISTANT_"))]
        cleared = {name: "" for name in names}
        return mock.patch.dict(os.environ, {**cleared, **values})

    def test_off_by_default(self) -> None:
        with self.env():
            config = IntentEngineConfig.from_env()
        self.assertFalse(config.enabled)
        self.assertEqual(config.blocked_domains, ("lock", "alarm_control_panel"))
        self.assertEqual(config.allowed_intents, DEFAULT_ALLOWED_INTENTS)
        self.assertEqual(config.sentences_dir, PACKAGED_SENTENCES_DIR)

    def test_home_assistant_needs_both_url_and_token(self) -> None:
        with self.env(INTENT_ENGINE_ENABLED="true", HOME_ASSISTANT_URL="http://ha.local:8123"):
            config = IntentEngineConfig.from_env()
        self.assertTrue(config.enabled)
        self.assertFalse(config.home_assistant_configured)
        self.assertFalse(config.snapshot()["home_assistant"])

    def test_blocked_domains_override(self) -> None:
        with self.env(INTENT_ENGINE_BLOCKED_DOMAINS="Lock, cover"):
            self.assertEqual(IntentEngineConfig.from_env().blocked_domains, ("lock", "cover"))
        with self.env(INTENT_ENGINE_BLOCKED_DOMAINS="none"):
            self.assertEqual(IntentEngineConfig.from_env().blocked_domains, ())

    def test_reads_all_settings_and_never_exposes_the_token(self) -> None:
        with self.env(
            INTENT_ENGINE_ENABLED="true",
            HOME_ASSISTANT_URL="http://ha.local:8123/",
            HOME_ASSISTANT_TOKEN="very-secret",
            INTENT_ENGINE_AREA="Salon",
            INTENT_ENGINE_DRY_RUN="true",
            INTENT_ENGINE_TIMEOUT_SECS="1.5",
            INTENT_ENGINE_WAKE_WORDS="Reachy, Richie ,",
            INTENT_ENGINE_ALLOWED_INTENTS="HassTurnOn,HassTurnOff",
        ):
            config = IntentEngineConfig.from_env()
        self.assertTrue(config.home_assistant_configured)
        self.assertEqual(config.home_assistant_url, "http://ha.local:8123")
        self.assertEqual((config.area, config.dry_run, config.timeout_secs), ("Salon", True, 1.5))
        self.assertEqual(config.wake_words, ("Reachy", "Richie"))
        self.assertEqual(config.allowed_intents, ("HassTurnOn", "HassTurnOff"))
        self.assertNotIn("very-secret", repr(config))
        snapshot = config.snapshot()
        self.assertNotIn("very-secret", str(snapshot))
        self.assertNotIn("ha.local", str(snapshot))
        self.assertEqual(snapshot["home_assistant"], True)
        self.assertEqual(snapshot["blocked_domains"], ["lock", "alarm_control_panel"])

    def test_default_allow_list_excludes_wildcard_and_timer_intents(self) -> None:
        for name in (
            "HassMediaSearchAndPlay",
            "HassBroadcast",
            "HassStartTimer",
            "HassListAddItem",
            "HassShoppingListAddItem",
        ):
            self.assertNotIn(name, DEFAULT_ALLOWED_INTENTS)


if __name__ == "__main__":
    unittest.main()

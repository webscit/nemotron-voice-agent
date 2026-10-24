# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Home Assistant adapter and response rendering, with response shapes recorded from a demo instance."""

import datetime
import json
import unittest

import httpx
from loguru import logger

from examples.shared.intent_engine import targets
from examples.shared.intent_engine.config import DEFAULT_ALLOWED_INTENTS, PACKAGED_SENTENCES_DIR
from examples.shared.intent_engine.matcher import IntentMatcher
from examples.shared.intent_engine.responses import coerce_speech_slots, is_local_template, render_local
from examples.shared.intent_engine.targets import HomeAssistantTarget, IntentRequest, TargetUnavailable

TOKEN = "long-lived-secret-token"
SYNC_PAYLOAD = {
    "entities": [
        {
            "id": "light.ceiling_lights",
            "name": "Ceiling Lights",
            "domain": "light",
            "device_class": None,
            "area": "salon",
        },
        {
            "id": "cover.kitchen_window",
            "name": "Kitchen Window",
            "domain": "cover",
            "device_class": "window",
            "area": None,
        },
    ],
    "areas": [{"id": "salon", "name": "Salon"}, {"id": "cuisine", "name": "Cuisine"}],
    "floors": [{"id": "rdc", "name": "Rez-de-chaussée"}],
}
# Recorded from a Home Assistant demo instance (2026-10).
ACTION_DONE = {
    "speech": {},
    "card": {},
    "language": "fr",
    "response_type": "action_done",
    "data": {"success": [{"name": "Ceiling Lights", "type": "entity", "id": "light.ceiling_lights"}], "failed": []},
}
QUERY_ANSWER = {**ACTION_DONE, "response_type": "query_answer"}
CURRENT_TIME = {
    "speech": {},
    "card": {},
    "language": "fr",
    "response_type": "action_done",
    "speech_slots": {"time": "11:53:44.860885"},
    "data": {"success": [], "failed": []},
}
MATCH_FAILED = {
    "speech": {
        "plain": {
            "speech": "<MatchFailedError result=MatchTargetsResult(is_match=False, "
            "no_match_reason=<MatchFailedReason.NAME: 1>, states=[], no_match_name=None, areas=[], floors=[]), "
            "constraints=MatchTargetsConstraints(name='Does Not Exist', assistant='conversation')>",
            "extra_data": None,
        }
    },
    "card": {},
    "language": "fr",
    "response_type": "error",
    "data": {"code": "failed_to_handle"},
}

MATCHER = IntentMatcher(
    "fr-FR",
    allowed_intents=DEFAULT_ALLOWED_INTENTS,
    sentences_dir=PACKAGED_SENTENCES_DIR,
    keep_custom_intent=lambda name: False,
)


class FakeHomeAssistant:
    """httpx transport that answers like the Home Assistant REST API and records requests."""

    def __init__(self, intent_response=None, *, status: int = 200, rendered: str = "Ceiling lights est on"):
        self.intent_response = intent_response if intent_response is not None else ACTION_DONE
        self.status = status
        self.rendered = rendered
        self.requests: list[tuple[str, dict, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body, request.headers.get("authorization", "")))
        if request.url.path == "/api/template":
            if "variables" in body:
                return httpx.Response(200, text=self.rendered)
            return httpx.Response(200, text=json.dumps(SYNC_PAYLOAD))
        if self.status != 200:
            return httpx.Response(self.status, text="500 Internal Server Error")
        return httpx.Response(200, json=self.intent_response)

    def paths(self) -> list[str]:
        return [path for path, _, _ in self.requests]


def make_target(server, **kwargs) -> HomeAssistantTarget:
    return HomeAssistantTarget(
        "http://ha.test/", TOKEN, templates=MATCHER, transport=httpx.MockTransport(server), **kwargs
    )


class HomeAssistantTargetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        targets._SYNC_CACHE.clear()

    async def test_sync_builds_slot_lists_from_one_template_call(self) -> None:
        server = FakeHomeAssistant()
        target = make_target(server, area="salon")
        lists = await target.sync()
        self.assertEqual(server.paths(), ["/api/template"])
        self.assertEqual(server.requests[0][2], f"Bearer {TOKEN}")
        self.assertEqual([v.value_out for v in lists["name"].values], ["Ceiling Lights", "Kitchen Window"])
        self.assertEqual(lists["name"].values[1].context, {"domain": "cover", "device_class": "window"})
        self.assertEqual([v.value_out for v in lists["area"].values], ["Salon", "Cuisine"])
        self.assertEqual([v.value_out for v in lists["floor"].values], ["Rez-de-chaussée"])
        self.assertIs(target.slot_lists, lists)

    async def test_sync_is_cached_across_sessions_until_ttl(self) -> None:
        server = FakeHomeAssistant()
        await make_target(server).sync()
        await make_target(server).sync()
        self.assertEqual(len(server.requests), 1)
        await make_target(server, cache_ttl_secs=0.0).sync()
        self.assertEqual(len(server.requests), 2)

    async def test_sync_failures_raise_target_unavailable(self) -> None:
        def unauthorized(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="401: Unauthorized")

        def garbage(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json")

        for handler in (unauthorized, garbage):
            with self.subTest(handler=handler.__name__), self.assertRaises(TargetUnavailable):
                await HomeAssistantTarget(
                    "http://ha.test", TOKEN, templates=MATCHER, transport=httpx.MockTransport(handler)
                ).sync()

    async def test_action_intent_posts_contract_and_renders_locally(self) -> None:
        server = FakeHomeAssistant()
        target = make_target(server, area="Salon")
        await target.sync()
        response = await target.handle(IntentRequest("HassTurnOn", {"name": "Ceiling Lights"}, "fr"))
        path, body, auth = server.requests[-1]
        self.assertEqual(path, "/api/intent/handle")
        self.assertEqual(
            body,
            {
                "name": "HassTurnOn",
                "data": {"name": "Ceiling Lights", "preferred_area_id": "salon"},
                "language": "fr",
                "assistant": "conversation",
            },
        )
        self.assertEqual(auth, f"Bearer {TOKEN}")
        self.assertEqual((response.speech, response.response_type), ("Allumé", "action_done"))
        self.assertEqual(response.data["success"][0]["id"], "light.ceiling_lights")
        self.assertEqual(server.paths().count("/api/template"), 1)  # only the sync

    async def test_area_wide_on_off_is_limited_to_unblocked_domains(self) -> None:
        def body_for(target, name, data):
            return target.build_body(IntentRequest(name, data, "fr"))["data"]

        blocked = make_target(FakeHomeAssistant(), blocked_domains=("lock", "cover"))
        await blocked.sync()
        self.assertEqual(body_for(blocked, "HassTurnOff", {"area": "Salon"}), {"area": "Salon", "domain": ["light"]})
        # Requests that already name a target or a domain, and other intents, are untouched.
        self.assertEqual(body_for(blocked, "HassTurnOff", {"name": "Ceiling Lights"}), {"name": "Ceiling Lights"})
        self.assertEqual(body_for(blocked, "HassTurnOn", {"area": "Salon", "domain": "light"})["domain"], "light")
        self.assertEqual(body_for(blocked, "HassMediaPause", {"area": "Salon"}), {"area": "Salon"})

        targets._SYNC_CACHE.clear()
        unrestricted = make_target(FakeHomeAssistant())
        await unrestricted.sync()
        self.assertEqual(body_for(unrestricted, "HassTurnOff", {"area": "Salon"}), {"area": "Salon"})

        targets._SYNC_CACHE.clear()
        server = FakeHomeAssistant()
        nothing_left = make_target(server, blocked_domains=("light", "cover"))
        await nothing_left.sync()
        with self.assertRaises(TargetUnavailable):
            await nothing_left.handle(IntentRequest("HassTurnOff", {"area": "Salon"}, "fr"))
        self.assertEqual(server.paths(), ["/api/template"])

    async def test_response_key_selects_template(self) -> None:
        target = make_target(FakeHomeAssistant())
        response = await target.handle(IntentRequest("HassTurnOn", {"name": "Kitchen Window"}, "fr", "cover"))
        self.assertEqual(response.speech, "Ouverture en cours")
        response = await target.handle(IntentRequest("HassClimateSetTemperature", {"temperature": 20}, "fr"))
        self.assertEqual(response.speech, "Température réglée sur 20 degrés")

    async def test_current_time_string_is_converted_for_the_template(self) -> None:
        target = make_target(FakeHomeAssistant(CURRENT_TIME))
        response = await target.handle(IntentRequest("HassGetCurrentTime", {}, "fr"))
        self.assertEqual(response.speech, "Il est 11:53")

    async def test_query_intent_is_rendered_on_home_assistant(self) -> None:
        server = FakeHomeAssistant(QUERY_ANSWER)
        target = make_target(server)
        response = await target.handle(IntentRequest("HassGetState", {"name": "Ceiling Lights"}, "fr", "one"))
        self.assertEqual((response.speech, response.response_type), ("Ceiling lights est on", "query_answer"))
        path, body, _ = server.requests[-1]
        self.assertEqual(path, "/api/template")
        self.assertEqual(body["variables"], {"slots": {"name": "Ceiling Lights"}})
        self.assertTrue(
            body["template"].startswith(
                "{% set query = {'matched': [states['light.ceiling_lights']], 'unmatched': []} %}"
                "{% set state = query.matched[0] %}"
            )
        )
        self.assertTrue(body["template"].endswith(MATCHER.response_template("HassGetState", "one")))

    async def test_query_without_usable_entity_is_unavailable(self) -> None:
        for success in (
            [],
            [{"name": "x", "type": "entity", "id": "light.x'] }}{{ evil"}],
            [{"type": "area", "id": "a"}],
        ):
            payload = {**QUERY_ANSWER, "data": {"success": success, "failed": []}}
            server = FakeHomeAssistant(payload)
            with self.subTest(success=success), self.assertRaises(TargetUnavailable):
                await make_target(server).handle(IntentRequest("HassGetState", {"name": "x"}, "fr", "one"))
            self.assertEqual(server.paths(), ["/api/intent/handle"])

    async def test_error_response_never_speaks_returned_speech(self) -> None:
        target = make_target(FakeHomeAssistant(MATCH_FAILED))
        response = await target.handle(IntentRequest("HassTurnOn", {"name": "Does Not Exist"}, "fr"))
        self.assertTrue(response.is_error)
        self.assertEqual(response.data["code"], "failed_to_handle")
        self.assertEqual(response.speech, "Une erreur est intervenue pendant le traitement")
        self.assertNotIn("MatchFailedError", response.speech)

    async def test_unknown_intent_http_500_is_unavailable(self) -> None:
        with self.assertRaises(TargetUnavailable) as raised:
            await make_target(FakeHomeAssistant(status=500)).handle(IntentRequest("HassNotAnIntent", {}, "fr"))
        self.assertIn("HTTP 500", str(raised.exception))

    async def test_connection_errors_and_timeouts_are_unavailable_and_token_is_not_logged(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        def too_slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        lines: list[str] = []
        sink = logger.add(lines.append, level="TRACE")
        try:
            for handler in (refuse, too_slow):
                target = HomeAssistantTarget(
                    "http://ha.test", TOKEN, templates=MATCHER, transport=httpx.MockTransport(handler)
                )
                with self.subTest(handler=handler.__name__), self.assertRaises(TargetUnavailable) as raised:
                    await target.handle(IntentRequest("HassTurnOn", {"name": "Ceiling Lights"}, "fr"))
                self.assertNotIn(TOKEN, str(raised.exception))
                self.assertNotIn(TOKEN, str(target.build_body(IntentRequest("HassTurnOn", {}, "fr"))))
        finally:
            logger.remove(sink)
        self.assertNotIn(TOKEN, "".join(lines))


class ResponseRenderingTests(unittest.TestCase):
    def test_local_versus_home_assistant_templates(self) -> None:
        self.assertTrue(is_local_template("Allumé"))
        self.assertTrue(is_local_template(MATCHER.response_template("HassGetCurrentTime")))
        self.assertTrue(is_local_template(MATCHER.response_template("HassGetCurrentDate")))
        for intent, key in (
            ("HassGetState", "one"),
            ("HassClimateGetTemperature", "default"),
            ("HassGetWeather", "default"),
        ):
            with self.subTest(intent=intent):
                self.assertFalse(is_local_template(MATCHER.response_template(intent, key)))
        self.assertFalse(is_local_template("{{ slots.name | some_home_assistant_filter }}"))
        self.assertFalse(is_local_template("{% broken"))

    def test_coerce_time_and_date_strings(self) -> None:
        slots = coerce_speech_slots({"time": "11:53:44.860885", "date": "2026-10-03", "name": "x"})
        self.assertEqual(slots["time"], datetime.time(11, 53, 44, 860885))
        self.assertEqual(slots["date"], datetime.date(2026, 10, 3))
        self.assertEqual(coerce_speech_slots({"time": "soon"})["time"], "soon")

    def test_render_local_normalizes_whitespace_and_merges_slots(self) -> None:
        self.assertEqual(render_local("  {{ slots.item }}\n ajouté \n", {"item": "lait"}), "lait ajouté")
        self.assertEqual(
            render_local(MATCHER.response_template("HassGetCurrentDate"), {"date": "2026-10-01"}),
            "Nous sommes le premier octobre 2026",
        )

    def test_templates_run_in_a_sandbox(self) -> None:
        from jinja2.exceptions import SecurityError

        with self.assertRaises(SecurityError):
            render_local("{{ slots.__class__.__mro__ }}", {})


if __name__ == "__main__":
    unittest.main()

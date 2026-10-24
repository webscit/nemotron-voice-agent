# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

"""Intent engine against a throwaway Home Assistant demo instance.

Opt-in: start ``docker/docker-compose.ha-dev.yaml`` and set
``RUN_HA_INTENT_INTEGRATION=1``. The tests switch demo entities on and off, so
they refuse any Home Assistant that is not on the loopback interface.
"""

import asyncio
import json
import os
import re
from urllib.parse import urlparse

import httpx
import pytest

from examples.shared.intent_engine import targets
from examples.shared.intent_engine.config import (
    DEFAULT_ALLOWED_INTENTS,
    DEFAULT_BLOCKED_DOMAINS,
    PACKAGED_SENTENCES_DIR,
)
from examples.shared.intent_engine.matcher import IntentMatcher
from examples.shared.intent_engine.targets import HomeAssistantTarget, IntentRequest, TargetUnavailable

HA_URL = os.getenv("HA_INTENT_TEST_URL", "http://127.0.0.1:8124").rstrip("/")
HA_USER = os.getenv("HA_INTENT_TEST_USER", "dev")
HA_PASSWORD = os.getenv("HA_INTENT_TEST_PASSWORD", "devdevdev")
AREA = "Salon"
EXPOSED_LIGHT = "light.ceiling_lights"
HIDDEN_LIGHT = "light.kitchen_lights"
# Defined in docker/ha-dev/configuration.yaml with a unique ID so it can live in an area.
AREA_LOCK = "lock.area_test_lock"

pytestmark = [
    pytest.mark.ha_integration,
    pytest.mark.skipif(
        os.getenv("RUN_HA_INTENT_INTEGRATION") != "1",
        reason="set RUN_HA_INTENT_INTEGRATION=1 with docker/docker-compose.ha-dev.yaml running",
    ),
    pytest.mark.skipif(
        urlparse(HA_URL).hostname not in {"127.0.0.1", "localhost", "::1"},
        reason="these tests change device states; only a loopback demo Home Assistant is allowed",
    ),
]


def _access_token() -> str:
    """Onboard a fresh demo instance, or log in to one that is already onboarded."""
    client_id = f"{HA_URL}/"
    with httpx.Client(base_url=HA_URL, timeout=30) as client:
        onboarding = client.post(
            "/api/onboarding/users",
            json={
                "client_id": client_id,
                "name": "Dev",
                "username": HA_USER,
                "password": HA_PASSWORD,
                "language": "fr",
            },
        )
        if onboarding.status_code == 200:
            code = onboarding.json()["auth_code"]
        else:
            flow = client.post(
                "/auth/login_flow",
                json={"client_id": client_id, "handler": ["homeassistant", None], "redirect_uri": client_id},
            ).json()
            code = client.post(
                f"/auth/login_flow/{flow['flow_id']}",
                json={"client_id": client_id, "username": HA_USER, "password": HA_PASSWORD},
            ).json()["result"]
        token = client.post(
            "/auth/token", data={"grant_type": "authorization_code", "code": code, "client_id": client_id}
        )
        token.raise_for_status()
        return token.json()["access_token"]


async def _prepare_registry(token: str) -> None:
    """Put one light in the agent's area and hide another from Assist (test setup only)."""
    import websockets

    async with websockets.connect(HA_URL.replace("http", "ws", 1) + "/api/websocket") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        assert json.loads(await ws.recv())["type"] == "auth_ok"
        counter = 0

        async def call(**command):
            nonlocal counter
            counter += 1
            await ws.send(json.dumps({"id": counter, **command}))
            while True:
                reply = json.loads(await ws.recv())
                if reply.get("id") == counter:
                    assert reply["success"], reply
                    return reply["result"]

        areas = await call(type="config/area_registry/list")
        area = next((a for a in areas if a["name"] == AREA), None) or await call(
            type="config/area_registry/create", name=AREA
        )
        await call(type="config/entity_registry/update", entity_id=EXPOSED_LIGHT, area_id=area["area_id"])
        registry = await call(type="config/entity_registry/list")
        if any(entry["entity_id"] == AREA_LOCK for entry in registry):
            await call(type="config/entity_registry/update", entity_id=AREA_LOCK, area_id=area["area_id"])
            await call(
                type="homeassistant/expose_entity",
                assistants=["conversation"],
                entity_ids=[AREA_LOCK],
                should_expose=True,
            )
        await call(
            type="homeassistant/expose_entity",
            assistants=["conversation"],
            entity_ids=[HIDDEN_LIGHT],
            should_expose=False,
        )


@pytest.fixture(scope="module")
def token() -> str:
    try:
        value = _access_token()
    except (httpx.HTTPError, KeyError) as exc:
        pytest.skip(f"demo Home Assistant at {HA_URL} is not usable: {exc!r}")
    asyncio.run(_prepare_registry(value))
    return value


@pytest.fixture(scope="module")
def matcher() -> IntentMatcher:
    return IntentMatcher(
        "fr-FR",
        allowed_intents=DEFAULT_ALLOWED_INTENTS,
        sentences_dir=PACKAGED_SENTENCES_DIR,
        wake_words=("Reachy",),
        area=AREA,
        keep_custom_intent=lambda name: False,
    )


def run(token: str, matcher: IntentMatcher, scenario, blocked_domains=DEFAULT_BLOCKED_DOMAINS):
    """Run ``scenario(target, say)`` against a freshly synced target."""

    async def main():
        targets._SYNC_CACHE.clear()
        target = HomeAssistantTarget(HA_URL, token, templates=matcher, area=AREA, blocked_domains=blocked_domains)
        lists = await target.sync()

        async def say(text: str):
            match = matcher.match(text, lists)
            assert match is not None, f"no match for {text!r}"
            return match, await target.handle(IntentRequest(match.name, match.slots, "fr", match.response_key))

        try:
            return await scenario(target, say)
        finally:
            await target.aclose()

    return asyncio.run(main())


def state_of(token: str, entity_id: str) -> str:
    response = httpx.get(f"{HA_URL}/api/states/{entity_id}", headers={"Authorization": f"Bearer {token}"})
    return response.json()["state"]


def test_sync_lists_demo_entities_and_area(token, matcher):
    async def scenario(target, say):
        names = {value.value_out: value.context for value in target.slot_lists["name"].values}
        assert names["Ceiling Lights"]["domain"] == "light"
        assert AREA in [value.value_out for value in target.slot_lists["area"].values]
        assert target._area_id

    run(token, matcher, scenario)


def test_turn_on_and_off_by_name(token, matcher):
    async def scenario(target, say):
        match, response = await say("Bonjour Reachy, allume Ceiling Lights")
        assert (match.name, response.response_type, response.speech) == ("HassTurnOn", "action_done", "Allumé")
        assert response.data["success"][0]["id"] == EXPOSED_LIGHT
        await asyncio.sleep(0.2)
        assert state_of(token, EXPOSED_LIGHT) == "on"
        match, response = await say("Est-ce que tu peux éteindre Ceiling Lights ?")
        assert (match.name, response.speech) == ("HassTurnOff", "Éteint")
        await asyncio.sleep(0.2)
        assert state_of(token, EXPOSED_LIGHT) == "off"

    run(token, matcher, scenario)


def test_area_context_targets_the_agent_area(token, matcher):
    async def scenario(target, say):
        match, response = await say("Allume les lumières")
        assert match.slots == {"domain": "light", "area": AREA}
        assert response.response_type == "action_done"
        assert [item["id"] for item in response.data["success"] if item["type"] == "entity"] == [EXPOSED_LIGHT]

    run(token, matcher, scenario)


def test_state_query_is_rendered_on_home_assistant(token, matcher):
    async def scenario(target, say):
        await say("Allume Ceiling Lights")
        await asyncio.sleep(0.2)
        match, response = await say("Quel est l'état de Ceiling Lights ?")
        assert (match.name, match.response_key, response.response_type) == ("HassGetState", "one", "query_answer")
        assert response.speech == "Ceiling lights est on"

    run(token, matcher, scenario)


def test_current_time_and_date_render_locally(token, matcher):
    async def scenario(target, say):
        _, response = await say("Quelle heure est-il ?")
        assert re.fullmatch(r"Il est \d\d:\d\d", response.speech)
        _, response = await say("Quel jour sommes-nous ?")
        assert re.fullmatch(r"Nous sommes le (premier|\d+) \S+ \d{4}", response.speech)

    run(token, matcher, scenario)


def test_unexposed_entity_is_refused_with_packaged_error(token, matcher):
    async def scenario(target, say):
        before = state_of(token, HIDDEN_LIGHT)
        match, response = await say("Éteins Kitchen Lights" if before == "on" else "Allume Kitchen Lights")
        assert response.response_type == "error"
        assert response.data["code"] == "failed_to_handle"
        assert response.speech == "Une erreur est intervenue pendant le traitement"
        await asyncio.sleep(0.2)
        assert state_of(token, HIDDEN_LIGHT) == before

    run(token, matcher, scenario)


def test_demo_locks_are_blocked_by_default(token, matcher):
    lock = "lock.front_door"

    async def blocked(target, say):
        domains = {value.context["domain"] for value in target.slot_lists["name"].values}
        assert "lock" not in domains and "alarm_control_panel" not in domains
        before = state_of(token, lock)
        for text in ("Déverrouille Front Door", "Verrouille Front Door", "Éteins Front Door", "Allume Front Door"):
            assert matcher.match(text, target.slot_lists) is None, text
        await asyncio.sleep(0.2)
        assert state_of(token, lock) == before
        return before

    async def unblocked(target, say):
        # Control: with the block lifted the same sentence does match a lock, so the
        # default block is what keeps it out.
        locks = [v.value_out for v in target.slot_lists["name"].values if v.context["domain"] == "lock"]
        assert len(locks) >= 4 and "Front Door" in locks
        match = matcher.match("Déverrouille Front Door", target.slot_lists)
        assert (match.name, match.slots, match.response_key) == ("HassTurnOff", {"name": "Front Door"}, "lock")

    run(token, matcher, blocked)
    run(token, matcher, unblocked, blocked_domains=())


def test_area_wide_on_off_never_reaches_a_lock_in_the_area(token, matcher):
    request = IntentRequest("HassTurnOff", {"area": AREA}, "fr")
    headers = {"Authorization": f"Bearer {token}"}
    if httpx.get(f"{HA_URL}/api/states/{AREA_LOCK}", headers=headers).status_code != 200:
        pytest.skip("recreate the container: its configuration predates the area test lock")

    def relock():
        httpx.post(f"{HA_URL}/api/services/lock/lock", headers=headers, json={"entity_id": AREA_LOCK})

    async def blocked(target, say):
        relock()
        await asyncio.sleep(0.5)
        body = target.build_body(request)
        assert "lock" not in body["data"]["domain"] and "light" in body["data"]["domain"]
        response = await target.handle(request)
        assert response.response_type == "action_done"
        assert EXPOSED_LIGHT in [item["id"] for item in response.data["success"]]
        await asyncio.sleep(1.0)
        assert state_of(token, AREA_LOCK) == "locked"

    async def unblocked(target, say):
        # Control: without the constraint Home Assistant unlocks the lock in the area.
        assert "domain" not in target.build_body(request)["data"]
        await target.handle(request)
        await asyncio.sleep(1.0)
        try:
            assert state_of(token, AREA_LOCK) == "unlocked"
        finally:
            relock()

    run(token, matcher, blocked)
    run(token, matcher, unblocked, blocked_domains=())


def test_unknown_intent_and_bad_token_are_unavailable(token, matcher):
    async def scenario(target, say):
        with pytest.raises(TargetUnavailable, match="HTTP 500"):
            await target.handle(IntentRequest("HassNotAnIntent", {}, "fr"))
        bad = HomeAssistantTarget(HA_URL, "not-a-token", templates=matcher)
        with pytest.raises(TargetUnavailable, match="HTTP 401"):
            await bad.handle(IntentRequest("HassTurnOn", {"name": "Ceiling Lights"}, "fr"))
        await bad.aclose()

    run(token, matcher, scenario)

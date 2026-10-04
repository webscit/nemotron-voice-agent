# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targets that execute a matched intent.

Every target takes an ``IntentRequest`` and returns an ``IntentResponse`` in
Home Assistant's intent-response shape, with ``speech`` already rendered:

- ``HomeAssistantTarget`` calls the Home Assistant REST API.
- ``ClientToolTarget`` forwards the intent to a tool the client declared, over
  the same RTVI function-call channel the LLM uses.

A target raises ``TargetUnavailable`` when it cannot answer; the engine then
hands the turn to the LLM.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx
import yaml
from hassil.intents import SlotList
from jinja2 import TemplateError
from loguru import logger
from pipecat.frames.frames import FunctionCallFromLLM, FunctionCallResultFrame, FunctionCallResultProperties
from pipecat.processors.frame_processor import FrameDirection

from examples.shared.intent_engine.matcher import build_slot_lists
from examples.shared.intent_engine.responses import is_local_template, render_local

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.llm_service import LLMService

RESPONSE_ACTION_DONE = "action_done"
RESPONSE_QUERY_ANSWER = "query_answer"
RESPONSE_ERROR = "error"

CLIENT_TOOLS_FILE = "client_tools.yaml"

_ENTITY_ID_RE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")

# One call returns everything the matcher needs: entity names with their domain and
# device class, plus area and floor names.
_SYNC_TEMPLATE = (
    "{% set ns = namespace(entities=[], areas=[], floors=[]) %}"
    "{% for s in states %}{% set ns.entities = ns.entities + [{"
    '"id": s.entity_id, "name": s.name, "domain": s.domain, '
    '"device_class": s.attributes.device_class | default(none), "area": area_id(s.entity_id)'
    "}] %}{% endfor %}"
    '{% for a in areas() %}{% set ns.areas = ns.areas + [{"id": a, "name": area_name(a)}] %}{% endfor %}'
    '{% for f in floors() %}{% set ns.floors = ns.floors + [{"id": f, "name": floor_name(f)}] %}{% endfor %}'
    '{{ {"entities": ns.entities, "areas": ns.areas, "floors": ns.floors} | tojson }}'
)

# Intents whose Home Assistant handler accepts any entity domain. A request with neither
# a name nor a domain makes it act on every exposed entity in the area, locks included.
_ANY_DOMAIN_INTENTS = frozenset({"HassTurnOn", "HassTurnOff"})

# Synced Home Assistant data shared by every session, keyed by base URL.
_SYNC_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class TargetUnavailable(Exception):
    """The target could not answer (unreachable, timeout, bad status, unusable reply)."""

    def __init__(self, message: str, *, llm_notified: bool = False):
        """Record the reason; ``llm_notified`` means the LLM already got the turn."""
        super().__init__(message)
        self.llm_notified = llm_notified


@dataclass(frozen=True)
class IntentRequest:
    """What the engine asks a target to do."""

    name: str
    data: dict[str, Any]
    language: str
    # Response template key of the matched sentence (``responses.intents[name][key]``).
    response_key: str = "default"


@dataclass
class IntentResponse:
    """Home Assistant intent-response shape with ``speech`` rendered by the target."""

    speech: str
    response_type: str = RESPONSE_ACTION_DONE
    speech_slots: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Return whether the target reported an error."""
        return self.response_type == RESPONSE_ERROR


class SpeechTemplates(Protocol):
    """Where targets look up response templates (implemented by ``IntentMatcher``)."""

    def response_template(self, match_or_name: str, response_key: str = "default") -> str | None:
        """Return the response template for an intent."""

    def error_template(self, key: str = "handle_error") -> str | None:
        """Return the localized error sentence."""


def _error_speech(templates: SpeechTemplates) -> str:
    """Return the packaged localized error sentence (never a target's own error text)."""
    template = templates.error_template("handle_error") or ""
    try:
        return render_local(template, {})
    except TemplateError:
        return ""


class HomeAssistantTarget:
    """Execute intents through the Home Assistant REST API with a long-lived token."""

    name = "home_assistant"

    def __init__(
        self,
        url: str,
        token: str,
        *,
        templates: SpeechTemplates,
        timeout_secs: float = 3.0,
        cache_ttl_secs: float = 60.0,
        area: str = "",
        blocked_domains: tuple[str, ...] = (),
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        """Create the target; no request is sent until ``sync`` or ``handle``."""
        self._url = url.rstrip("/")
        self._templates = templates
        self.timeout_secs = timeout_secs
        self._cache_ttl_secs = cache_ttl_secs
        self._area = area
        self._blocked_domains = blocked_domains
        self._area_id: str | None = None
        self._unblocked_domains: list[str] = []
        self.slot_lists: dict[str, SlotList] | None = None
        self._client = httpx.AsyncClient(
            base_url=self._url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_secs,
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        try:
            response = await self._client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise TargetUnavailable(f"{path}: {type(exc).__name__}") from exc
        if response.status_code != 200:
            # 401: bad token. 500: unknown intent name or a handler crash.
            raise TargetUnavailable(f"{path}: HTTP {response.status_code}")
        return response

    async def render_template(self, template: str, variables: dict[str, Any] | None = None) -> str:
        """Render a template on Home Assistant and return the text."""
        body: dict[str, Any] = {"template": template}
        if variables:
            body["variables"] = variables
        return (await self._post("/api/template", body)).text

    async def sync(self) -> dict[str, SlotList]:
        """Load entity, area, and floor names (one request, cached across sessions)."""
        cached = _SYNC_CACHE.get(self._url)
        if cached and time.monotonic() - cached[0] <= self._cache_ttl_secs:
            payload = cached[1]
        else:
            try:
                payload = json.loads(await self.render_template(_SYNC_TEMPLATE))
            except ValueError as exc:
                raise TargetUnavailable("entity sync returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise TargetUnavailable("entity sync returned an unexpected payload")
            _SYNC_CACHE[self._url] = (time.monotonic(), payload)

        areas = [area for area in payload.get("areas") or [] if isinstance(area, dict)]
        floors = [floor for floor in payload.get("floors") or [] if isinstance(floor, dict)]
        self._area_id = next(
            (area.get("id") for area in areas if str(area.get("name", "")).lower() == self._area.lower()), None
        )
        if self._area and self._area_id is None:
            logger.warning(f"Intent engine: area {self._area!r} is not a Home Assistant area name")
        entities = [entity for entity in payload.get("entities") or [] if isinstance(entity, dict)]
        self._unblocked_domains = sorted(
            {str(entity.get("domain")) for entity in entities if entity.get("domain")} - set(self._blocked_domains)
        )
        self.slot_lists = build_slot_lists(
            entities,
            [area.get("name") for area in areas],
            [floor.get("name") for floor in floors],
            self._blocked_domains,
        )
        return self.slot_lists

    def build_body(self, request: IntentRequest) -> dict[str, Any]:
        """Return the ``/api/intent/handle`` body for a request (also logged by dry-run)."""
        data = dict(request.data)
        if self._area_id and "preferred_area_id" not in data:
            # Same hint Home Assistant adds for a satellite: prefer entities in the agent's area.
            data["preferred_area_id"] = self._area_id
        if (
            self._blocked_domains
            and request.name in _ANY_DOMAIN_INTENTS
            and not data.get("name")
            and not data.get("domain")
        ):
            # "Turn everything off here": limit the request to the domains that are not
            # blocked, or Home Assistant would also unlock the locks in the area.
            data["domain"] = self._unblocked_domains
        # ``assistant`` makes Home Assistant enforce its Assist entity exposure settings.
        return {"name": request.name, "data": data, "language": request.language, "assistant": "conversation"}

    async def handle(self, request: IntentRequest) -> IntentResponse:
        """Execute the intent, then render its speech locally or on Home Assistant."""
        body = self.build_body(request)
        if body["data"].get("domain") == []:
            raise TargetUnavailable("no controllable domain is known for an area-wide request")
        try:
            payload = (await self._post("/api/intent/handle", body)).json()
        except ValueError as exc:
            raise TargetUnavailable("intent response is not JSON") from exc
        if not isinstance(payload, dict):
            raise TargetUnavailable("intent response is not an object")

        response = IntentResponse(
            speech="",
            response_type=str(payload.get("response_type") or RESPONSE_ACTION_DONE),
            speech_slots=payload.get("speech_slots") or {},
            data=payload.get("data") or {},
        )
        if response.is_error:
            # The returned speech is a raw Python error repr; say the packaged sentence instead.
            response.speech = _error_speech(self._templates)
            return response

        template = self._templates.response_template(request.name, request.response_key)
        if not template:
            return response
        slots = {**request.data, **response.speech_slots}
        try:
            if is_local_template(template):
                response.speech = render_local(template, slots)
            else:
                response.speech = await self._render_query(template, slots, response.data)
        except TemplateError as exc:
            raise TargetUnavailable(f"response template failed: {type(exc).__name__}") from exc
        return response

    async def _render_query(self, template: str, slots: dict[str, Any], data: dict[str, Any]) -> str:
        """Render a state-reading template on Home Assistant, bound to the returned targets."""
        entity_ids = [
            target.get("id")
            for target in data.get("success") or []
            if isinstance(target, dict) and target.get("type") == "entity"
        ]
        entity_ids = [entity_id for entity_id in entity_ids if _ENTITY_ID_RE.match(str(entity_id or ""))]
        if not entity_ids:
            raise TargetUnavailable("query returned no entity to describe")
        matched = ", ".join(f"states['{entity_id}']" for entity_id in entity_ids)
        prefix = f"{{% set query = {{'matched': [{matched}], 'unmatched': []}} %}}{{% set state = query.matched[0] %}}"
        rendered = await self.render_template(prefix + template, {"slots": slots})
        return " ".join(rendered.split())


class ClientToolMap:
    """Intent -> client tool mapping, loaded from ``client_tools.yaml``.

    Each entry names the ``tool`` to call and its ``arguments``. An argument is
    either a literal or ``{slot: <slot name>, default: <value>}``. An entry with
    ``defer_to_llm: true`` reserves its sentences for the LLM.

    An entry can chain a second call to the same tool under ``then``. Its
    arguments can also be ``{result: <key>, add: <literal or slot reference>,
    min: <n>, max: <n>}``, computed from the first call's result. This is how a
    relative change is expressed for a tool that only takes absolute values:
    read the current value, then set the new one.
    """

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None):
        """Wrap parsed mapping entries keyed by intent name."""
        self._entries = {name: entry for name, entry in (entries or {}).items() if isinstance(entry, dict)}

    @classmethod
    def load(cls, sentences_dir: Path) -> ClientToolMap:
        """Load the mapping file from a sentences directory (missing file -> empty map)."""
        path = sentences_dir / CLIENT_TOOLS_FILE
        if not path.is_file():
            return cls()
        try:
            content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(f"Intent engine: ignoring unreadable {path}: {exc}")
            return cls()
        return cls(content.get("intents") if isinstance(content, dict) else None)

    def __contains__(self, intent_name: str) -> bool:
        """Return whether an intent is routed to a client tool."""
        return intent_name in self._entries

    def tool(self, intent_name: str) -> str:
        """Return the client tool name for an intent."""
        return str(self._entries[intent_name].get("tool") or "")

    def defers_to_llm(self, intent_name: str) -> bool:
        """Return whether a matched intent must be answered by the LLM."""
        return bool(self._entries[intent_name].get("defer_to_llm"))

    def has_follow_up(self, intent_name: str) -> bool:
        """Return whether the entry chains a second call under ``then``."""
        return isinstance(self._entries[intent_name].get("then"), dict)

    def arguments(self, intent_name: str, slots: dict[str, Any]) -> dict[str, Any]:
        """Build the tool arguments from the matched slots."""
        return self._build(self._entries[intent_name].get("arguments"), slots, None)

    def follow_up_arguments(self, intent_name: str, slots: dict[str, Any], result: Any) -> dict[str, Any]:
        """Build the second call's arguments from the slots and the first call's result.

        Raises ``ValueError`` when the result lacks a number the mapping needs.
        """
        return self._build(self._entries[intent_name]["then"].get("arguments"), slots, result)

    @classmethod
    def _build(cls, specs: Any, slots: dict[str, Any], result: Any) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        for key, spec in (specs or {}).items():
            value = cls._resolve(spec, slots, result)
            if value is not None:
                arguments[key] = value
        return arguments

    @classmethod
    def _resolve(cls, spec: Any, slots: dict[str, Any], result: Any) -> Any:
        if not isinstance(spec, dict):
            return spec
        if "slot" in spec:
            return slots.get(spec["slot"], spec.get("default"))
        if "result" in spec:
            base = result.get(spec["result"]) if isinstance(result, dict) else None
            offset = cls._resolve(spec.get("add", 0), slots, result)
            numbers = (int, float)
            if isinstance(base, bool) or not isinstance(base, numbers) or not isinstance(offset, numbers):
                raise ValueError(f"result has no numeric {spec['result']!r}")
            value = base + offset
            if "min" in spec:
                value = max(spec["min"], value)
            if "max" in spec:
                value = min(spec["max"], value)
            return int(value) if float(value).is_integer() else value
        return spec


class ClientToolTarget:
    """Run an intent as a client-declared tool call.

    ``LLMService.run_function_calls`` broadcasts the same frames an LLM tool call
    does, so RTVI forwards the call to the client and the assistant aggregator
    records the call and its result in the context. The client's result comes
    back as a ``FunctionCallResultFrame`` travelling downstream from the RTVI
    processor; the engine shows it to ``observe_result``, which marks it
    ``run_llm=False`` so the aggregator does not ask the LLM to comment on it.
    """

    name = "client_tool"

    def __init__(
        self,
        llm: LLMService,
        context: LLMContext,
        tool_map: ClientToolMap,
        *,
        templates: SpeechTemplates,
        timeout_secs: float,
    ):
        """Create the target; ``timeout_secs`` bounds each client call."""
        self._llm = llm
        self._context = context
        self._tool_map = tool_map
        self._templates = templates
        self._call_timeout_secs = timeout_secs
        # An intent is at most two chained calls.
        self.timeout_secs = 2 * timeout_secs + 0.5
        self._pending: dict[str, asyncio.Future] = {}

    def build_call(self, request: IntentRequest) -> tuple[str, dict[str, Any]]:
        """Return the ``(tool, arguments)`` a request maps to (also logged by dry-run)."""
        return self._tool_map.tool(request.name), self._tool_map.arguments(request.name, request.data)

    def observe_result(self, frame: FunctionCallResultFrame, direction: FrameDirection) -> None:
        """Resolve a pending call when its result frame passes through the engine."""
        future = self._pending.get(frame.tool_call_id)
        if future is None or future.done():
            return
        if direction == FrameDirection.DOWNSTREAM:
            # The client's answer. Keep it out of the LLM: the engine speaks the reply.
            if frame.properties is None:
                frame.properties = FunctionCallResultProperties(run_llm=False)
            else:
                frame.properties.run_llm = False
            future.set_result(frame.result)
        else:
            # Upstream copy of a result broadcast by the LLM service itself: the
            # registered handler timed out. Its downstream copy already re-triggers
            # the LLM, so this turn belongs to the LLM.
            future.set_exception(TargetUnavailable("client did not answer the tool call", llm_notified=True))

    def cancel_pending(self) -> None:
        """Forget in-flight calls (the user interrupted or the session ended)."""
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()

    async def _call(self, tool: str, arguments: dict[str, Any]) -> tuple[str, Any]:
        """Run one client tool call and return ``(tool_call_id, result)``."""
        tool_call_id = f"intent-{uuid.uuid4().hex[:16]}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[tool_call_id] = future
        try:
            await self._llm.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name=tool, tool_call_id=tool_call_id, arguments=arguments, context=self._context
                    )
                ]
            )
            return tool_call_id, await asyncio.wait_for(future, timeout=self._call_timeout_secs)
        finally:
            self._pending.pop(tool_call_id, None)

    def _error(self, tool: str, tool_call_id: str) -> IntentResponse:
        return IntentResponse(
            speech=_error_speech(self._templates),
            response_type=RESPONSE_ERROR,
            data={
                "code": "failed_to_handle",
                "success": [],
                "failed": [{"name": tool, "type": "tool", "id": tool_call_id}],
            },
        )

    @staticmethod
    def _failed(result: Any) -> bool:
        return isinstance(result, dict) and bool(result.get("error") or result.get("status") == "error")

    async def handle(self, request: IntentRequest) -> IntentResponse:
        """Call the mapped client tool, wait for its result, and render the reply."""
        tool, arguments = self.build_call(request)
        tool_call_id, result = await self._call(tool, arguments)
        if self._failed(result):
            return self._error(tool, tool_call_id)
        if self._tool_map.has_follow_up(request.name):
            try:
                arguments = self._tool_map.follow_up_arguments(request.name, request.data, result)
            except ValueError as exc:
                logger.warning(f"Intent engine: {request.name} cannot chain its second call: {exc}")
                return self._error(tool, tool_call_id)
            tool_call_id, result = await self._call(tool, arguments)
            if self._failed(result):
                return self._error(tool, tool_call_id)

        # The template sees the matched slots plus the arguments of the last call.
        speech_slots = {**request.data, **arguments}
        template = self._templates.response_template(request.name, request.response_key) or ""
        try:
            speech = render_local(template, speech_slots)
        except TemplateError as exc:
            logger.warning(f"Intent engine: response template for {request.name} failed: {exc}")
            speech = ""
        return IntentResponse(
            speech=speech,
            speech_slots=speech_slots,
            data={"success": [{"name": tool, "type": "tool", "id": tool_call_id}], "failed": []},
        )

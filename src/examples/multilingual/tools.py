# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Client-declared tool schemas for the multilingual assistant pipeline.

Every tool executed here runs on the *client*, not the server — the browser
is the only party in a position to do so safely or accurately (it alone
knows the user's own wall-clock time, and arithmetic is evaluated inside a
sandboxed Web Worker with every network-capable API disabled, so the server
never has to run arbitrary LLM-authored code).

Rather than a fixed schema baked into the server, the client declares what
it can do at connect time: ``requestData.tools`` on the WebRTC offer (or the
``client_tools`` query parameter on the WebSocket URL) carries a list of
``{name, description, parameters}`` entries, one per
``client.registerFunctionCallHandler`` the browser has wired up (see
``client/src/lib/clientTools.ts``). ``build_client_tools`` below turns that
untrusted, client-supplied JSON into a ``ToolsSchema`` the LLM context can
use, forwarding each declared name to the client via RTVI (see
``tool_handlers.py``) exactly like a server-defined tool would be forwarded.

Since the payload arrives over the wire from the browser, it is validated as
a system boundary: malformed or excessive entries are dropped with a
warning rather than allowed to reach the LLM or crash the session.
"""

from __future__ import annotations

import re

from loguru import logger
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_DESCRIPTION_LEN = 2000
_MAX_TOOLS = 20


def _validate_tool(raw: object) -> dict | None:
    """Return a well-formed OpenAI-style function schema for one declared tool, or None."""
    if not isinstance(raw, dict):
        logger.warning(f"Ignoring client tool declaration that is not an object: {raw!r}")
        return None

    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        logger.warning(f"Ignoring client tool declaration with invalid name: {name!r}")
        return None

    description = raw.get("description", "")
    if not isinstance(description, str) or len(description) > _MAX_DESCRIPTION_LEN:
        logger.warning(f"Ignoring client tool '{name}': invalid or oversized description")
        return None

    parameters = raw.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        logger.warning(f"Ignoring client tool '{name}': parameters must be a JSON-schema object")
        return None

    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def build_client_tools(raw_tools: object) -> tuple[ToolsSchema | None, tuple[str, ...]]:
    """Validate a client-declared tool list and build the schema the LLM context uses.

    Returns ``(None, ())`` when there is nothing usable to offer — callers
    should leave the context's ``tools`` as ``NOT_GIVEN`` in that case rather
    than passing an empty ``ToolsSchema``.
    """
    if not isinstance(raw_tools, list):
        if raw_tools:
            logger.warning(f"Ignoring client tool declarations: expected a list, got {type(raw_tools).__name__}")
        return None, ()

    schemas: list[dict] = []
    names: list[str] = []
    for raw in raw_tools[:_MAX_TOOLS]:
        schema = _validate_tool(raw)
        if schema is None:
            continue
        name = schema["function"]["name"]
        if name in names:
            logger.warning(f"Ignoring duplicate client tool declaration: {name}")
            continue
        schemas.append(schema)
        names.append(name)

    if len(raw_tools) > _MAX_TOOLS:
        logger.warning(f"Client declared {len(raw_tools)} tools, only the first {_MAX_TOOLS} were considered")

    if not schemas:
        return None, ()

    return ToolsSchema(standard_tools=[], custom_tools={AdapterType.OPENAI: schemas}), tuple(names)

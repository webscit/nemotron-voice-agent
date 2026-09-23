# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Client-executed tool schemas for the multilingual assistant pipeline.

Each tool here is executed on the *client*, not the server, because the
client is the only party in a position to do so safely or accurately:

- ``get_client_local_time``: only the browser knows the user's own
  wall-clock time and IANA timezone.
- ``evaluate_math_expression``: arithmetic is evaluated inside a sandboxed
  Web Worker in the browser (see ``client/src/lib/mathWorkerClient.ts``) with
  every network-capable API disabled, so the server never runs arbitrary
  LLM-authored code.

See ``tool_handlers.py`` for how the server forwards these calls over RTVI
and waits for the client's answer.
"""

from __future__ import annotations

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema

GET_CLIENT_LOCAL_TIME_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "get_client_local_time",
        "description": (
            "Get the user's current local date, time, and timezone directly from their "
            "own device. Use this whenever the user asks what time, date, or day it is "
            "for them, since the server may run in a different timezone."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}

EVALUATE_MATH_EXPRESSION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "evaluate_math_expression",
        "description": (
            "Evaluate a math expression (arithmetic, powers, roots, trigonometry, logarithms) "
            "using a sandboxed calculator in the user's browser. Use this for any calculation "
            "instead of computing or guessing the answer yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A plain arithmetic expression using +, -, *, /, %, ^ (power), "
                        "parentheses, numeric literals, and Math function names such as "
                        "sqrt, pow, sin, cos, log, abs (for example 'sqrt(16) + 2^3')."
                    ),
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

CLIENT_TOOLS_SCHEMA = ToolsSchema(
    standard_tools=[],
    custom_tools={AdapterType.OPENAI: [GET_CLIENT_LOCAL_TIME_TOOL, EVALUATE_MATH_EXPRESSION_TOOL]},
)

CLIENT_TOOL_NAMES: tuple[str, ...] = ("get_client_local_time", "evaluate_math_expression")

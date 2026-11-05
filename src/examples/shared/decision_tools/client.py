# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for the Ollama System One decision API (``POST /v1/systemone``).

One request asks one ``choice`` question per tool and returns, for each, the
most likely option with its probability distribution and confidence. See
https://docs.ollama.com/capabilities/decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from examples.shared.decision_tools.config import DecisionModelConfig, DecisionTool


class DecisionError(Exception):
    """The decision model did not answer usably (unreachable, timeout, bad status or body)."""


@dataclass(frozen=True)
class Decision:
    """The decision model's answer for one tool."""

    tool: str
    choice: str
    confidence: float
    probabilities: dict[str, float]


class SystemOneClient:
    """Ask the decision model which value each tool should take."""

    def __init__(self, config: DecisionModelConfig, *, transport: httpx.AsyncBaseTransport | None = None):
        """Create the client; ``transport`` replaces the network in tests."""
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self._config = config
        self._client = httpx.AsyncClient(timeout=config.timeout_secs, headers=headers, transport=transport)

    async def decide(
        self, state: str, tools: tuple[DecisionTool, ...], *, timeout_secs: float | None = None
    ) -> list[Decision]:
        """Return one decision per tool the model answered for ``state``.

        ``timeout_secs`` replaces the configured timeout for this request.
        """
        body = {
            "model": self._config.model,
            "state": state,
            "questions": {tool.name: tool.question() for tool in tools},
        }
        try:
            response = await self._client.post(
                self._config.url,
                json=body,
                timeout=httpx.USE_CLIENT_DEFAULT if timeout_secs is None else timeout_secs,
            )
            response.raise_for_status()
            answers = response.json().get("answers")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise DecisionError(f"{type(exc).__name__}: {exc}") from exc
        if not isinstance(answers, dict):
            raise DecisionError("response has no 'answers' object")
        return [decision for tool in tools if (decision := _parse_answer(tool, answers.get(tool.name)))]

    async def close(self) -> None:
        """Close the HTTP connection pool."""
        await self._client.aclose()


def _parse_answer(tool: DecisionTool, answer: Any) -> Decision | None:
    """Read one ``choice`` answer; None when it is missing or malformed."""
    if not isinstance(answer, dict) or not isinstance(answer.get("choice"), str):
        return None
    probabilities = answer.get("probabilities")
    try:
        confidence = float(answer.get("confidence", 0.0))
        probabilities = {str(k): float(v) for k, v in probabilities.items()} if isinstance(probabilities, dict) else {}
    except (TypeError, ValueError):
        return None
    return Decision(tool=tool.name, choice=answer["choice"], confidence=confidence, probabilities=probabilities)

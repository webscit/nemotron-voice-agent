# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settings and tool catalog for the decision-controlled client tools."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from utils import parse_env_float

PACKAGED_TOOLS_FILE = Path(__file__).resolve().parent / "decision_tools.yaml"

# The option the decision model picks when no value fits; never sent to the client.
NONE_CHOICE = "none"
# A System One ``choice`` question takes 2 to 26 options, ``none`` included.
MAX_CHOICES = 25


@dataclass(frozen=True)
class DecisionModelConfig:
    """The decision model endpoint (Ollama System One); empty ``url`` turns the feature off."""

    url: str = ""
    model: str = "nimble"
    # Never log or snapshot the key; ``repr=False`` keeps it out of dataclass reprs.
    api_key: str = field(default="", repr=False)
    # A 25-value question is ~900 prompt tokens: ~1.3 s for nimble on an idle Jetson
    # Thor, more while the LLM and TTS share the GPU.
    timeout_secs: float = 3.0

    @classmethod
    def from_env(cls) -> DecisionModelConfig:
        """Read the ``DECISION_MODEL_*`` variables."""
        return cls(
            url=(os.getenv("DECISION_MODEL_URL") or "").strip(),
            model=(os.getenv("DECISION_MODEL") or "").strip() or "nimble",
            api_key=(os.getenv("DECISION_MODEL_API_KEY") or "").strip(),
            timeout_secs=parse_env_float("DECISION_MODEL_TIMEOUT_SECS", 3.0, min_value=0.1),
        )

    @property
    def enabled(self) -> bool:
        """Return whether a decision model endpoint is configured."""
        return bool(self.url)

    def snapshot(self) -> dict:
        """Describe the decision model for the recorded session snapshot (no key)."""
        return {"enabled": self.enabled, "model": self.model, "timeout_secs": self.timeout_secs}


@dataclass(frozen=True)
class DecisionTool:
    """A client tool whose call the decision model decides after each LLM answer."""

    name: str
    argument: str
    instructions: str
    none: str
    criteria: dict[str, str]
    min_confidence: float = 0.0
    arguments: dict[str, Any] = field(default_factory=dict)

    def question(self) -> dict[str, Any]:
        """Return the System One ``choice`` question for this tool."""
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": {**self.criteria, NONE_CHOICE: self.none},
        }

    def call_arguments(self, choice: str) -> dict[str, Any]:
        """Return the client tool arguments for a chosen value."""
        return {**self.arguments, self.argument: choice}


def _parse_tool(name: str, raw: Any) -> DecisionTool:
    """Validate one catalog entry; raises ``ValueError`` on a malformed entry."""
    if not isinstance(raw, dict):
        raise ValueError("entry must be a mapping")
    argument = raw.get("argument")
    instructions = raw.get("instructions")
    none = raw.get("none")
    criteria = raw.get("criteria")
    arguments = raw.get("arguments") or {}
    if not isinstance(argument, str) or not argument:
        raise ValueError("'argument' must be a non-empty string")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("'instructions' must be a non-empty string")
    if not isinstance(none, str) or not none.strip():
        raise ValueError("'none' must be a non-empty string")
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("'criteria' must be a non-empty mapping")
    if len(criteria) > MAX_CHOICES:
        raise ValueError(f"'criteria' has {len(criteria)} values, the decision model accepts {MAX_CHOICES}")
    if NONE_CHOICE in criteria:
        raise ValueError(f"'{NONE_CHOICE}' is reserved for the no-value option")
    if not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in criteria.items()):
        raise ValueError("'criteria' must map each value to a description")
    if not isinstance(arguments, dict):
        raise ValueError("'arguments' must be a mapping")
    try:
        min_confidence = float(raw.get("min_confidence", 0.0))
    except (TypeError, ValueError):
        raise ValueError("'min_confidence' must be a number") from None
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("'min_confidence' must be between 0 and 1")
    return DecisionTool(
        name=name,
        argument=argument,
        instructions=" ".join(instructions.split()),
        none=none.strip(),
        criteria={k: v.strip() for k, v in criteria.items()},
        min_confidence=min_confidence,
        arguments=dict(arguments),
    )


def load_decision_tools(path: Path = PACKAGED_TOOLS_FILE) -> dict[str, DecisionTool]:
    """Load the catalog of decision-controlled tools; malformed entries are skipped with a warning."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(f"Decision tools: cannot read {path}: {exc}")
        return {}
    raw_tools = data.get("tools") if isinstance(data, dict) else None
    if not isinstance(raw_tools, dict):
        logger.warning(f"Decision tools: {path} has no 'tools' mapping")
        return {}
    tools: dict[str, DecisionTool] = {}
    for name, raw in raw_tools.items():
        try:
            tools[str(name)] = _parse_tool(str(name), raw)
        except ValueError as exc:
            logger.warning(f"Decision tools: ignoring '{name}': {exc}")
    return tools


def _declared_enum(raw_tool: dict, argument: str) -> list | None:
    """Return the ``enum`` the client declared for ``argument``, or None when it has none."""
    parameters = raw_tool.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    spec = properties.get(argument) if isinstance(properties, dict) else None
    enum = spec.get("enum") if isinstance(spec, dict) else None
    return enum if isinstance(enum, list) else None


def active_decision_tools(
    tools: dict[str, DecisionTool], raw_client_tools: object, client_tool_names: tuple[str, ...]
) -> tuple[DecisionTool, ...]:
    """Return the catalog tools this client declared, restricted to the values its schema accepts.

    ``client_tool_names`` are the declarations ``build_client_tools`` accepted;
    ``raw_client_tools`` is the client's payload, read for each tool's ``enum``.
    """
    raw_by_name = (
        {raw.get("name"): raw for raw in raw_client_tools if isinstance(raw, dict) and isinstance(raw.get("name"), str)}
        if isinstance(raw_client_tools, list)
        else {}
    )
    active: list[DecisionTool] = []
    for name in client_tool_names:
        tool = tools.get(name)
        if tool is None:
            continue
        enum = _declared_enum(raw_by_name.get(name) or {}, tool.argument)
        if enum is None:
            active.append(tool)
            continue
        criteria = {value: text for value, text in tool.criteria.items() if value in enum}
        dropped = sorted(set(tool.criteria) - set(criteria))
        if dropped:
            logger.info(f"Decision tools: '{name}' does not accept {', '.join(dropped)}")
        if criteria:
            active.append(replace(tool, criteria=criteria))
        else:
            logger.warning(f"Decision tools: '{name}' accepts none of the configured values; left to the LLM")
    return tuple(active)

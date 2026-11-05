# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``redecide``: ask other decision models the questions of the live decision tools.

For each user turn where the live decision model ran (``decision_tool`` events of
the multilingual example's decision tools):

1. replay the turn's first decision (the same ``state`` and the same questions,
   taken from the session snapshot) on the configured ``reference`` model and on
   every ``candidates`` model, storing ``kind="decision"`` annotations
   (source = endpoint name);
2. pick the reference choice: the latest human choice (``kind="decision"``,
   ``source="human:<name>"``) wins over the reference model;
3. score the live choice (``source="live:<decision model>"``) and each candidate
   against it: one ``kind="decision_summary"`` per session and source, with the
   number of turns that agree.

Endpoints are Ollama System One URLs (``POST /v1/systemone``). An endpoint
without ``url`` uses ``DECISION_MODEL_URL`` (the live decision model's server).
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from monitoring.jobs.base import Job, JobContext, register

DECISION_EVENT = "decision_tool"
DECISION_KIND = "decision"
SUMMARY_KIND = "decision_summary"
_TIMEOUT_SECS = 120.0  # also covers loading the model on the first request


@dataclass(frozen=True)
class DecisionEndpoint:
    """One decision model the job asks."""

    name: str
    model: str
    url: str
    api_key: str = ""
    service: str | None = None

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> DecisionEndpoint:
        """Build from a ``redecide`` entry; ``url`` defaults to ``DECISION_MODEL_URL``."""
        model = str(raw.get("model") or "")
        return cls(
            name=str(raw.get("name") or model),
            model=model,
            url=str(raw.get("url") or os.getenv("DECISION_MODEL_URL") or "").strip(),
            api_key=(os.getenv(str(raw.get("api_key_env") or "DECISION_MODEL_API_KEY")) or "").strip(),
            service=raw.get("service"),
        )

    def decide(self, client: httpx.Client, state: str, questions: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Return ``{tool: {choice, confidence, probabilities}}`` for the answered questions."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = client.post(
            self.url, json={"model": self.model, "state": state, "questions": questions}, headers=headers
        )
        response.raise_for_status()
        answers = response.json().get("answers") or {}
        return {
            tool: {
                "choice": answer["choice"],
                "confidence": answer.get("confidence"),
                "probabilities": answer.get("probabilities") or {},
            }
            for tool, answer in answers.items()
            if tool in questions and isinstance(answer, dict) and isinstance(answer.get("choice"), str)
        }


def redecide_endpoints(config: dict[str, Any]) -> tuple[DecisionEndpoint | None, list[DecisionEndpoint]]:
    """Enabled ``redecide`` reference and candidates (``enabled: false`` or no URL skips one)."""
    section = config.get("redecide") or {}
    raw_reference = section.get("reference")
    endpoints = [
        DecisionEndpoint.from_config(raw) if raw and raw.get("enabled", True) else None
        for raw in [raw_reference, *(section.get("candidates") or [])]
    ]
    usable = [e if e and e.url and e.model else None for e in endpoints]
    return usable[0], [e for e in usable[1:] if e]


def redecide_names(config: dict[str, Any]) -> tuple[str | None, list[str]]:
    """Names of the enabled reference and candidates, whether or not their URL is known here.

    The review server scores human choices with them; it does not call the models.
    """
    section = config.get("redecide") or {}

    def name(raw: dict[str, Any] | None) -> str | None:
        if not raw or not raw.get("enabled", True):
            return None
        return str(raw.get("name") or raw.get("model") or "") or None

    candidates = [name(raw) for raw in section.get("candidates") or []]
    return name(section.get("reference")), [c for c in candidates if c]


def live_source_name(session_config: dict[str, Any]) -> str:
    """Annotation source name of the decisions made during the live session."""
    return f"live:{(session_config.get('decision_tools') or {}).get('model') or 'decision'}"


def first_decisions(store, session_id: str) -> dict[int, dict[str, Any]]:
    """``{turn idx: {"state", "choices": {tool: choice}, "confidences": {tool: confidence}}}`` of each first answer."""
    turns: dict[int, dict[str, Any]] = {}
    for event in store.rows("events", session_id, kind=DECISION_EVENT):
        data = event.get("data") or {}
        if event["turn_idx"] is None or not data.get("state"):
            continue
        turn = turns.setdefault(event["turn_idx"], {"state": data["state"], "choices": {}, "confidences": {}})
        if data["state"] != turn["state"]:
            continue  # a later answer of the same turn
        if data.get("tool") and data.get("choice"):
            turn["choices"].setdefault(data["tool"], data["choice"])
            if data.get("confidence") is not None:
                turn["confidences"].setdefault(data["tool"], data["confidence"])
    return turns


@register
class RedecideJob(Job):
    """Replay the live decisions on other decision models and score agreement."""

    kind = "redecide"

    def required_services(self, ctx: JobContext) -> list[str]:
        """Containers of the configured endpoints, when the session ran decision tools."""
        if not self._questions(ctx):
            return []
        reference, candidates = redecide_endpoints(ctx.config)
        return [e.service for e in [reference, *candidates] if e and e.service]

    @staticmethod
    def _questions(ctx: JobContext) -> dict[str, Any]:
        session = ctx.store.get_session(ctx.job["target"]) or {}
        return ((session.get("config") or {}).get("decision_tools") or {}).get("questions") or {}

    def run(self, ctx: JobContext) -> None:
        """Ask the missing (turn, endpoint) decisions, then (re)compute agreement."""
        session_id = ctx.job["target"]
        questions = self._questions(ctx)
        if not questions:
            logger.info(f"redecide {session_id}: the session ran no decision tools")
            return
        reference, candidates = redecide_endpoints(ctx.config)
        reference_name, candidate_names = redecide_names(ctx.config)
        usable = {e.name for e in [reference, *candidates] if e}
        skipped = [name for name in [reference_name, *candidate_names] if name and name not in usable]
        if skipped:
            logger.warning(
                f"redecide {session_id}: skipping {', '.join(skipped)}: no URL. Set the endpoint's `url`, or "
                "DECISION_MODEL_URL in the dreamer's environment (recreate the container after changing .env)"
            )
        turns = first_decisions(ctx.store, session_id)

        # Skip (endpoint, turn) pairs already asked, by this run (checkpoint) or an earlier one.
        done = {tuple(item) for item in ctx.progress.get("done", [])}
        for row in ctx.store.annotations_for(session_id, kind=DECISION_KIND):
            done.add((row["source"], int(row["target_id"].rsplit(":", 1)[1])))
        with httpx.Client(timeout=_TIMEOUT_SECS) as client:
            for endpoint in [e for e in [reference, *candidates] if e]:
                for turn_idx, turn in sorted(turns.items()):
                    if (endpoint.name, turn_idx) in done:
                        continue
                    ctx.check_preempted()
                    # The questions of the tools the live model answered (all of them after an error).
                    asked = {tool: q for tool, q in questions.items() if tool in turn["choices"]} or questions
                    answers = endpoint.decide(client, turn["state"], asked)
                    ctx.store.add_annotations(
                        [
                            {
                                "session_id": session_id,
                                "target_type": "turn",
                                "target_id": f"{session_id}:{turn_idx}",
                                "source": endpoint.name,
                                "source_version": endpoint.model,
                                "kind": DECISION_KIND,
                                "value": {"answers": answers},
                            }
                        ]
                    )
                    done.add((endpoint.name, turn_idx))
                    ctx.save_progress(done=sorted(done))

        summaries = score_session(
            ctx.store,
            session_id,
            reference_name=reference.name if reference else None,
            candidate_names=[c.name for c in candidates],
        )
        if not summaries:
            logger.info(
                f"redecide {session_id}: no reference decision available (no reference answer, no human choice)"
            )
        for summary in summaries:
            value = summary["value"]
            logger.info(f"redecide {session_id}: {summary['source']} agrees on {value['agreed']}/{value['turns']}")


def turn_answers(store, session_id: str) -> dict[tuple[int, str], dict[str, dict[str, Any]]]:
    """``{(turn idx, source): {tool: answer}}`` from the ``decision`` annotations.

    Later annotations of a source override its earlier answers tool by tool, so a
    reviewer who sets one tool, then another, keeps both.
    """
    merged: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in store.annotations_for(session_id, kind=DECISION_KIND):
        answers = (row["value"] or {}).get("answers") or {}
        merged[(int(row["target_id"].rsplit(":", 1)[1]), row["source"])].update(
            {tool: answer for tool, answer in answers.items() if isinstance(answer, dict) and answer.get("choice")}
        )
    return dict(merged)


def turn_choices(store, session_id: str) -> dict[tuple[int, str], dict[str, str]]:
    """``{(turn idx, source): {tool: choice}}`` from the ``decision`` annotations."""
    return {
        key: {tool: answer["choice"] for tool, answer in answers.items()}
        for key, answers in turn_answers(store, session_id).items()
    }


def latest_human_sources(store, session_id: str) -> dict[int, str]:
    """``{turn idx: human source}`` of the most recent human choice per turn."""
    latest: dict[int, str] = {}
    for row in store.annotations_for(session_id, kind=DECISION_KIND):
        if row["source"].startswith("human:"):
            latest[int(row["target_id"].rsplit(":", 1)[1])] = row["source"]
    return latest


def score_session(
    store, session_id: str, *, reference_name: str | None, candidate_names: list[str]
) -> list[dict[str, Any]]:
    """(Re)compute the ``decision_summary`` annotations of a session; returns them.

    A turn agrees when every tool got the reference's choice. CPU-only and fast.
    """
    session = store.get_session(session_id) or {}
    live_source = live_source_name(session.get("config") or {})
    choices = turn_choices(store, session_id)
    for turn_idx, turn in first_decisions(store, session_id).items():
        if turn["choices"]:
            choices[(turn_idx, live_source)] = turn["choices"]
    latest_human = latest_human_sources(store, session_id)

    totals: dict[str, dict[str, Any]] = defaultdict(lambda: {"agreed": 0, "turns": 0, "refs": set()})
    for turn_idx in sorted({idx for idx, _ in choices}):
        ref_source = latest_human.get(turn_idx) or reference_name
        reference = choices.get((turn_idx, ref_source)) if ref_source else None
        if not reference:
            continue
        for source in (live_source, *candidate_names):
            hypothesis = choices.get((turn_idx, source))
            if hypothesis is None:
                continue
            compared = [tool for tool in hypothesis if tool in reference]
            if not compared:
                continue
            total = totals[source]
            total["turns"] += 1
            total["agreed"] += all(hypothesis[tool] == reference[tool] for tool in compared)
            total["refs"].add(ref_source)

    summaries = [
        {
            "session_id": session_id,
            "target_type": "session",
            "target_id": session_id,
            "source": source,
            "source_version": None,
            "kind": SUMMARY_KIND,
            "value": {
                "agreement": total["agreed"] / total["turns"] if total["turns"] else None,
                "agreed": total["agreed"],
                "turns": total["turns"],
                "reference_sources": sorted(total["refs"]),
            },
        }
        for source, total in totals.items()
    ]
    store.replace_annotations(session_id, (SUMMARY_KIND,), summaries)
    return summaries

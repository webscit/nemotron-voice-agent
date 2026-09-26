# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare recorded sessions grouped by pipeline variant.

Groups sessions by fields of their config snapshot (default: LLM, ASR and TTS
models plus language) and prints, per group, latency percentiles (user→bot,
per-service TTFB, LLM TTFT split text-only vs. with images), interruption rate
and ASR WER from ``reasr`` annotations.

Usage::

    uv run python -m monitoring.report
    uv run python -m monitoring.report --since-days 7 --group-by llm.model,turn_detection.silero_vad_only
    uv run python -m monitoring.report --session 3f2a9c1d0b7e --csv report.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import select

from monitoring import schema
from monitoring.config import load_monitoring_config
from monitoring.store import SessionStore

DEFAULT_GROUP_BY = "llm.model,asr.model,tts.model,language"


def _dig(config: dict[str, Any], dotted: str) -> str:
    value: Any = config
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return "" if value is None else str(value)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(sorted(values), n=100, method="inclusive")[round(q) - 1]


def _fmt(value: float | None, unit: str = "s") -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%" if unit == "%" else f"{value:.3f}{unit}"


def _service(processor: str | None) -> str:
    return re.sub(r"#\d+$", "", processor or "?")


def collect(store: SessionStore, *, since: float | None, session_ids: list[str] | None, group_by: list[str]):
    """Return ``{group_key: stats}`` for the selected sessions."""
    with store.engine.connect() as conn:
        stmt = select(schema.sessions)
        if since:
            stmt = stmt.where(schema.sessions.c.started_at >= since)
        if session_ids:
            stmt = stmt.where(schema.sessions.c.id.in_(session_ids))
        sessions = [dict(r) for r in conn.execute(stmt).mappings()]
        ids = [s["id"] for s in sessions]

        def rows(table, *columns):
            if not ids:
                return []
            query = select(*(table.c[c] for c in columns)).where(table.c.session_id.in_(ids))
            return [dict(r) for r in conn.execute(query).mappings()]

        metrics = rows(schema.metrics, "session_id", "processor", "name", "value")
        turns = rows(schema.turns, "session_id", "idx", "user_text", "interrupted")
        llm_calls = rows(schema.llm_calls, "session_id", "ttfb", "n_images", "prompt_tokens", "completion_tokens")
        wers = [
            a
            for a in rows(schema.annotations, "session_id", "kind", "source", "value", "id")
            if a["kind"] == "wer_summary"
        ]

    group_of = {s["id"]: tuple(_dig(s["config"] or {}, g) for g in group_by) for s in sessions}
    groups: dict[tuple, dict[str, Any]] = defaultdict(
        lambda: {
            "sessions": 0,
            "turns": 0,
            "interrupted": 0,
            "latency": [],
            "first_latency": [],
            "ttfb": defaultdict(list),
            "ttft_text": [],
            "ttft_vision": [],
            "prompt_tokens": [],
            "wer": defaultdict(lambda: [0, 0]),
        }
    )
    for session in sessions:
        groups[group_of[session["id"]]]["sessions"] += 1
    for turn in turns:
        if turn["user_text"]:
            g = groups[group_of[turn["session_id"]]]
            g["turns"] += 1
            g["interrupted"] += bool(turn["interrupted"])
    for metric in metrics:
        g = groups[group_of[metric["session_id"]]]
        if metric["name"] == "user_bot_latency":
            g["latency"].append(metric["value"])
        elif metric["name"] == "first_bot_speech_latency":
            g["first_latency"].append(metric["value"])
        elif metric["name"] == "ttfb":
            g["ttfb"][_service(metric["processor"])].append(metric["value"])
    for call in llm_calls:
        g = groups[group_of[call["session_id"]]]
        if call["ttfb"] is not None:
            (g["ttft_vision"] if call["n_images"] else g["ttft_text"]).append(call["ttfb"])
        if call["prompt_tokens"]:
            g["prompt_tokens"].append(call["prompt_tokens"])
    latest_wer: dict[tuple[str, str], dict] = {}
    for annotation in sorted(wers, key=lambda a: a["id"]):
        latest_wer[(annotation["session_id"], annotation["source"])] = annotation["value"] or {}
    for (session_id, source), value in latest_wer.items():
        acc = groups[group_of[session_id]]["wer"][source]
        acc[0] += value.get("word_errors") or 0
        acc[1] += value.get("ref_words") or 0
    return groups


def summarize(groups, group_by: list[str]) -> list[dict[str, str]]:
    """Flatten grouped stats into printable rows."""
    rows = []
    for key, g in sorted(groups.items()):
        row = {name: value or "-" for name, value in zip(group_by, key, strict=True)}
        row.update(
            {
                "sessions": str(g["sessions"]),
                "turns": str(g["turns"]),
                "interrupt_rate": _fmt(g["interrupted"] / g["turns"] if g["turns"] else None, "%"),
                "user_bot_p50": _fmt(_pct(g["latency"], 50)),
                "user_bot_p90": _fmt(_pct(g["latency"], 90)),
                "first_speech_p50": _fmt(_pct(g["first_latency"], 50)),
                "llm_ttft_text_p50": _fmt(_pct(g["ttft_text"], 50)),
                "llm_ttft_vision_p50": _fmt(_pct(g["ttft_vision"], 50)),
                "prompt_tokens_p50": str(int(_pct(g["prompt_tokens"], 50))) if g["prompt_tokens"] else "-",
            }
        )
        for service, values in sorted(g["ttfb"].items()):
            row[f"ttfb_p50[{service}]"] = _fmt(_pct(values, 50))
            row[f"ttfb_p90[{service}]"] = _fmt(_pct(values, 90))
        for source, (errors, words) in sorted(g["wer"].items()):
            row[f"wer[{source}]"] = _fmt(errors / words if words else None, "%")
        rows.append(row)
    return rows


def main() -> int:
    """Entry point."""
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group-by", default=DEFAULT_GROUP_BY, help=f"config fields (default: {DEFAULT_GROUP_BY})")
    parser.add_argument("--since-days", type=float, help="only sessions started in the last N days")
    parser.add_argument("--session", action="append", help="restrict to session id(s)")
    parser.add_argument("--csv", help="also write the table to this CSV file")
    args = parser.parse_args()

    group_by = [g.strip() for g in args.group_by.split(",") if g.strip()]
    store = SessionStore(load_monitoring_config().db_url)
    since = time.time() - args.since_days * 86400 if args.since_days else None
    rows = summarize(collect(store, since=since, session_ids=args.session, group_by=group_by), group_by)
    if not rows:
        print("No recorded sessions match (is MONITORING_ENABLED=true?).")
        return 0
    for i, row in enumerate(rows, 1):
        print(f"── group {i} " + "─" * 60)
        width = max(len(k) for k in row)
        for key, value in row.items():
            print(f"  {key:<{width}}  {value}")
    if args.csv:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, restval="-")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip user-turn transcripts through a CSV for human correction.

``export`` writes one row per recorded user turn (audio path, live transcript and
every model transcript); fix the ``corrected`` column in any spreadsheet, then
``import`` stores non-empty corrections as ``source="human:<annotator>"``
transcripts, which ``reasr`` uses as the WER reference. Re-queue the sessions
afterwards (``python -m monitoring.jobs enqueue reasr <ids>``) to rescore.

Usage::

    uv run python -m monitoring.transcripts_csv export turns.csv [--session ID ...]
    uv run python -m monitoring.transcripts_csv import turns.csv --annotator alice
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from monitoring.config import load_monitoring_config
from monitoring.store import SessionStore


def export(store: SessionStore, path: str, session_ids: list[str] | None, artifacts_dir: Path) -> int:
    """Write the correction sheet; returns the number of rows."""
    ids = session_ids or store.ended_session_ids()
    count = 0
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["session_id", "turn_idx", "language", "audio", "live", "models", "corrected"])
        for session_id in ids:
            audio = defaultdict(list)
            for row in store.rows("media", session_id, modality="audio_user"):
                audio[row["turn_idx"]].append(str(artifacts_dir / row["artifact_key"]))
            models = defaultdict(dict)
            corrected = {}
            for row in store.annotations_for(session_id, kind="transcript"):
                text = (row["value"] or {}).get("text", "")
                if row["source"].startswith("human:"):
                    corrected[row["target_id"]] = text
                else:
                    models[row["target_id"]][row["source"]] = text
            for turn in store.rows("turns", session_id):
                if turn["idx"] not in audio:
                    continue
                target = f"{session_id}:{turn['idx']}"
                model_text = " | ".join(f"{src}: {text}" for src, text in sorted(models[target].items()))
                writer.writerow(
                    [
                        session_id,
                        turn["idx"],
                        turn["language"] or "",
                        " ".join(audio[turn["idx"]]),
                        turn["user_text"] or "",
                        model_text,
                        corrected.get(target, ""),
                    ]
                )
                count += 1
    return count


def import_(store: SessionStore, path: str, annotator: str) -> int:
    """Store non-empty ``corrected`` cells; returns the number imported."""
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("corrected") or "").strip()
            if not text:
                continue
            session_id = row["session_id"]
            rows.append(
                {
                    "session_id": session_id,
                    "target_type": "turn",
                    "target_id": f"{session_id}:{int(row['turn_idx'])}",
                    "source": f"human:{annotator}",
                    "kind": "transcript",
                    "value": {"text": text, "language": row.get("language") or None},
                }
            )
    store.add_annotations(rows)
    return len(rows)


def main() -> int:
    """Entry point."""
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("csv")
    exp.add_argument("--session", action="append")
    imp = sub.add_parser("import")
    imp.add_argument("csv")
    imp.add_argument("--annotator", required=True)
    args = parser.parse_args()

    settings = load_monitoring_config()
    store = SessionStore(settings.db_url)
    if args.command == "export":
        print(f"Exported {export(store, args.csv, args.session, settings.artifacts_dir)} turn(s) to {args.csv}")
    else:
        print(f"Imported {import_(store, args.csv, args.annotator)} correction(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

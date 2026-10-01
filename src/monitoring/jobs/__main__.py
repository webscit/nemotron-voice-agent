# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dreamer CLI.

Usage (from the repo root, ``PYTHONPATH=src`` or inside the container)::

    python -m monitoring.jobs run                      # idle-gated loop (compose `dreamer`)
    python -m monitoring.jobs once                     # one poll, then exit
    python -m monitoring.jobs enqueue reasr <session>  # (re)queue a job from scratch
    python -m monitoring.jobs status                   # job table summary
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from dotenv import load_dotenv

from monitoring.config import load_monitoring_config
from monitoring.jobs import JOB_REGISTRY
from monitoring.jobs.runner import Dreamer, load_dreamer_config
from monitoring.store import open_stores


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(prog="python -m monitoring.jobs", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="dreamer.yaml path (default: DREAMER_CONFIG or the bundled file)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run forever, only while no conversation is live")
    sub.add_parser("once", help="run at most one job and exit")
    enqueue = sub.add_parser("enqueue", help="queue a job from scratch (replaces a previous run)")
    enqueue.add_argument("kind", choices=sorted(JOB_REGISTRY))
    enqueue.add_argument("session_ids", nargs="+")
    sub.add_parser("status", help="show job counts and failures")
    args = parser.parse_args(argv)

    store, artifacts = open_stores(load_monitoring_config())
    config = load_dreamer_config(args.config)

    if args.command == "run":
        Dreamer(store, artifacts, config).run_forever()
    elif args.command == "once":
        dreamer = Dreamer(store, artifacts, config)
        dreamer.startup()
        print("ran a job" if dreamer.run_once() else "nothing to do (or not idle)")
    elif args.command == "enqueue":
        for session_id in args.session_ids:
            store.reset_job(args.kind, session_id)
        print(f"queued {args.kind} for {len(args.session_ids)} session(s)")
    elif args.command == "status":
        jobs = store.jobs()
        for (kind, status), count in sorted(Counter((j["kind"], j["status"]) for j in jobs).items()):
            print(f"{kind:12s} {status:8s} {count}")
        for job in jobs:
            if job["status"] == "failed" or job["error"]:
                print(f"  {job['kind']}#{job['id']} {job['target']} [{job['status']}]: {job['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

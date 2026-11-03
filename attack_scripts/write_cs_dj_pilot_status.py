#!/usr/bin/env python3
"""Atomically update the persistent CS-DJ pilot status file."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--state", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--detail", default="")
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    destination = args.run_dir / "pipeline_status.json"
    previous = {}
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding="utf-8"))
    payload = {
        **previous,
        "state": args.state,
        "stage": args.stage,
        "detail": args.detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "writer_pid": os.getpid(),
    }
    if "started_at" not in payload:
        payload["started_at"] = payload["updated_at"]
    if args.exit_code is not None:
        payload["exit_code"] = args.exit_code
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


if __name__ == "__main__":
    main()

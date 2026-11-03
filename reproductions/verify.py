#!/usr/bin/env python3
"""Verify the published evidence without loading models or making API calls."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path = ROOT) -> dict:
    root = root.resolve()
    catalog = json.loads((root / "reproductions/experiments.json").read_text())
    inventory = json.loads((root / "reproductions/files.json").read_text())["files"]
    paths = set()
    for entry in inventory:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or entry["path"] in paths:
            raise ValueError(f"Invalid or duplicate archive path: {entry['path']}")
        paths.add(entry["path"])
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {entry['path']}")
        if path.suffix == ".json":
            json.loads(data)
        elif path.suffix == ".csv":
            rows = list(csv.reader(data.decode().splitlines()))
            if not rows or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError(f"Malformed CSV: {entry['path']}")
    ids = set()
    for experiment in catalog["experiments"]:
        if experiment["id"] in ids:
            raise ValueError(f"Duplicate experiment: {experiment['id']}")
        ids.add(experiment["id"])
        if experiment.get("code_status") == "removed":
            if experiment.get("runner") or not experiment.get("historical_runner") or not experiment.get("code_removed_on"):
                raise ValueError(f"Incomplete removed-code record: {experiment['id']}")
        elif not experiment.get("runner") or not (root / experiment["runner"]).is_file():
            raise ValueError(f"Missing runner: {experiment.get('runner')}")
        if not experiment["evidence"] or not set(experiment["evidence"]) <= paths:
            raise ValueError(f"Missing evidence: {experiment['id']}")
    return {"experiments": len(ids), "verified_files": len(paths), "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))

#!/usr/bin/env python3
"""Restore a historical runner's layout; never execute it or replace local files."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from verify import ROOT, verify

DEPENDENCIES = {
    "jps_train_20261024": ["jps_paper_20261024"],
    "ultrabreak_train_20261024": ["ultrabreak_released_20261024"],
}


def restore(recipe: str, destination: Path, root: Path = ROOT) -> int:
    verify(root)
    source = root / "reproductions/run_sources/runs"
    available = {p.name for p in source.iterdir() if p.is_dir()}
    if recipe not in available:
        raise ValueError(f"Unknown recipe: {recipe}")
    recipes = [recipe, *DEPENDENCIES.get(recipe, [])]
    destination = destination.resolve()
    copies = []
    for name in recipes:
        for path in sorted((source / name).rglob("*")):
            if not path.is_file():
                continue
            target = destination / "runs" / path.relative_to(source)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Destination escapes workspace: {target}")
            if target.exists():
                if not target.is_file() or target.read_bytes() != path.read_bytes():
                    raise FileExistsError(f"Existing local file differs; nothing copied: {target}")
            else:
                copies.append((path, target))
    # Check every conflict before writing any file.
    for path, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return len(copies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", help="Name of an archived run directory")
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        names = sorted(p.name for p in (ROOT / "reproductions/run_sources/runs").iterdir() if p.is_dir())
        print("\n".join(names))
        return
    if not args.recipe:
        parser.error("use --list or --recipe NAME")
    count = restore(args.recipe, args.destination)
    print(json.dumps({"copied_files": count, "executed": False, "destination": str(args.destination)}))


if __name__ == "__main__":
    main()

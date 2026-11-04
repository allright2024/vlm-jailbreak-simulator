"""Move a legacy data bundle into benchmarks/ and method_assets/.

Usage: python -m core.migrate_data_layout --data-root /path/to/data --dry-run
The command moves files without rewriting benchmark contents, model inputs,
or historical run records. Existing destination data is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .paths import BENCHMARK_DIRS, METHOD_ASSET_DIRS, DATA_ROOT


def migration_plan(root: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    moves: list[tuple[Path, Path]] = []
    aliases: list[Path] = []
    known = BENCHMARK_DIRS | METHOD_ASSET_DIRS | {"benchmarks", "method_assets"}
    unknown = sorted(p.name for p in root.iterdir() if p.name not in known)
    if unknown:
        raise ValueError(f"Unclassified data-root entries; classify these first: {unknown}")
    for name in sorted((BENCHMARK_DIRS | METHOD_ASSET_DIRS) - {"AdvBenchM"}):
        source = root / name
        if not source.exists() and not source.is_symlink():
            continue
        section = "benchmarks" if name in BENCHMARK_DIRS else "method_assets"
        moves.append((source, root / section / name))
    old = root / "AdvBenchM"
    if old.is_dir():
        split = (old / "original_release").is_dir() or (old / "jood_subset").is_dir()
        if not split:
            # Original Drive bundles contain only JOOD's flat subset here.
            moves.append((old, root / "method_assets" / "JOOD"))
        else:
            for name, destination in [
                ("original_release", root / "benchmarks" / "AdvBenchM"),
                ("jood_subset", root / "method_assets" / "JOOD"),
            ]:
                if (old / name).exists():
                    moves.append((old / name, destination))
            for source in sorted(old.iterdir()):
                if source.name in {"original_release", "jood_subset"}:
                    continue
                if source.is_symlink() and os.readlink(source).startswith("jood_subset/"):
                    aliases.append(source)
                else:
                    moves.append((source, root / "method_assets" / "JOOD" /
                                  "provenance" / "previous_AdvBenchM" / source.name))
    destinations = [destination for _, destination in moves]
    if len(destinations) != len(set(destinations)):
        raise FileExistsError("Multiple legacy folders map to the same destination; resolve them first.")
    for source, destination in moves:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {destination} while moving {source}")
    return moves, aliases


def migrate_data_root(root: Path, *, dry_run: bool = False) -> dict:
    root = root.expanduser().resolve()
    moves, aliases = migration_plan(root)
    report = {
        "layout_version": 2,
        "data_root": str(root),
        "moves": [{"from": str(a.relative_to(root)), "to": str(b.relative_to(root))}
                  for a, b in moves],
        "removed_legacy_aliases": {str(p.relative_to(root)): os.readlink(p) for p in aliases},
        "benchmark_contents_modified": False,
        "existing_AdvBenchM_loader": "JOOD subset, 170 questions; unchanged",
    }
    if dry_run or not moves and not aliases:
        return report
    # Preflight has checked every destination before the first move.
    completed: list[tuple[Path, Path]] = []
    try:
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            completed.append((source, destination))
    except BaseException:
        for source, destination in reversed(completed):
            source.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(source)
        raise
    for alias in aliases:
        alias.unlink()
    old = root / "AdvBenchM"
    if old.is_dir() and not any(old.iterdir()):
        old.rmdir()
    for section in ["benchmarks", "method_assets"]:
        (root / section).mkdir(exist_ok=True)
    record = root / "method_assets" / "provenance" / "data_layout_migration.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(DATA_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = migrate_data_root(args.data_root, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

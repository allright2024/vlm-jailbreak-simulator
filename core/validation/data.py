#!/usr/bin/env python3
"""Validate an external VLM-Jailbreak-Simulator data directory.

This command never downloads or modifies data. It validates the Drive layout,
dataset row counts, and every image referenced by the dataset loaders.
"""

from __future__ import annotations

import argparse
import os


def _image_path(path: str, data_root: str, repo_root: str) -> str:
    """Resolve references against the selected root, not import-time globals."""
    value = os.path.expanduser(os.fspath(path))
    if os.path.isabs(value):
        return os.path.abspath(value)
    normalized = os.path.normpath(value)
    if normalized == "data":
        return data_root
    prefix = "data" + os.sep
    if normalized.startswith(prefix):
        return os.path.join(data_root, normalized[len(prefix):])
    return os.path.join(repo_root, normalized)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        help="External data directory (also accepted via VLM_JAILBREAK_DATA_ROOT).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to validate; repeatable. Default: all datasets.",
    )
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Only check required files/directories; do not parse all rows/images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # ``python -m`` imports core before parsing these arguments. Honor the
    # explicit root locally instead of mutating already-imported core globals.
    import core.datasets  # noqa: F401
    from core import DATA_ROOT, REPO_ROOT, get_dataset
    from core.data import DATASET_LAYOUT, EXPECTED_COUNTS

    data_root = (
        os.path.abspath(os.path.expanduser(args.data_root))
        if args.data_root else DATA_ROOT
    )

    datasets = args.datasets or sorted(DATASET_LAYOUT)
    unknown = sorted(set(datasets) - set(DATASET_LAYOUT))
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}")

    missing = sorted({
        os.path.join(data_root, relative)
        for name in datasets for relative in DATASET_LAYOUT[name]
        if not os.path.exists(os.path.join(data_root, relative))
    })
    if missing:
        print(f"Data root: {data_root}")
        print("Missing required paths:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(2)

    print(f"Data root: {data_root}")
    if args.layout_only:
        print(f"OK: layout present for {', '.join(datasets)}")
        return

    total = 0
    for name in datasets:
        dataset_class = get_dataset(name)
        dataset_dir = os.path.join(data_root, dataset_class.dir_name or dataset_class.name)
        behaviors = dataset_class(data_dir=dataset_dir).load()
        expected = EXPECTED_COUNTS[name]
        if len(behaviors) != expected:
            raise SystemExit(f"{name}: {len(behaviors)} rows, expected {expected}")
        ids = [behavior.behavior_id for behavior in behaviors]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{name}: duplicate behavior_id")
        missing_images = [
            path
            for behavior in behaviors
            for path in behavior.images
            if not os.path.isfile(_image_path(path, data_root, REPO_ROOT))
        ]
        if missing_images:
            raise SystemExit(
                f"{name}: {len(missing_images)} missing referenced images; "
                f"first={missing_images[0]}"
            )
        total += len(behaviors)
        print(f"  {name:15s} rows={len(behaviors):4d} images={sum(len(b.images) for b in behaviors):5d}")
    print(f"OK: {total} behaviors across {len(datasets)} datasets")


if __name__ == "__main__":
    main()

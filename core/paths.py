"""Resolve benchmark inputs and method assets below a shared data root.

New bundles use ``benchmarks/`` and ``method_assets/``. Legacy flat bundles
and paths in historical configurations remain readable without mixing the
original AdvBench-M release with JOOD's separate 170-question subset.
"""

from __future__ import annotations

import os
from os import PathLike
from typing import Union

Pathish = Union[str, PathLike[str]]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT_ENV = "VLM_JAILBREAK_DATA_ROOT"
BENCHMARK_DIRS = frozenset({
    "advbench", "AdvBenchM", "HADES", "MM-SafetyBench", "SafeBench", "HarmBench",
})
METHOD_ASSET_DIRS = frozenset({
    "CS-DJ", "JOOD", "MML", "nltk_data", "VisualAdv", "JailbreakInPieces",
    "JPS", "UltraBreak", "FORCE", "legacy",
})


def _absolute(path: Pathish) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


DATA_ROOT = _absolute(os.environ.get(DATA_ROOT_ENV, os.path.join(REPO_ROOT, "data")))


def _layout_candidates(relative: Pathish) -> list[str]:
    """Return the canonical path followed by compatible legacy locations."""
    normalized = os.path.normpath(os.fspath(relative))
    parts = normalized.split(os.sep)
    first, tail = parts[0], parts[1:]
    if first == "benchmarks" and tail:
        name, rest = tail[0], tail[1:]
        legacy = ["AdvBenchM", "original_release", *rest] if name == "AdvBenchM" else tail
        return [normalized, os.path.join(*legacy)]
    if first == "method_assets" and tail:
        name, rest = tail[0], tail[1:]
        if name == "JOOD":
            return [normalized, os.path.join("AdvBenchM", "jood_subset", *rest),
                    os.path.join("AdvBenchM", *rest), os.path.join(*tail)]
        return [normalized, os.path.join(*tail)]
    if first == "AdvBenchM":
        if tail and tail[0] == "original_release":
            return [os.path.join("benchmarks", "AdvBenchM", *tail[1:]), normalized]
        if tail and tail[0] == "jood_subset":
            rest = tail[1:]
        else:
            rest = tail
        # A legacy bare AdvBenchM path means the historical JOOD subset.
        return [os.path.join("method_assets", "JOOD", *rest),
                os.path.join("AdvBenchM", "jood_subset", *rest),
                os.path.join("AdvBenchM", *rest)]
    if first in BENCHMARK_DIRS:
        return [os.path.join("benchmarks", first, *tail), normalized]
    if first in METHOD_ASSET_DIRS:
        return [os.path.join("method_assets", first, *tail), normalized]
    return [normalized]


def resolve_data_path(relative: Pathish, data_root: Pathish | None = None) -> str:
    """Prefer the separated layout, falling back to existing flat bundles."""
    root = _absolute(data_root) if data_root is not None else DATA_ROOT
    candidates = [os.path.join(root, path) for path in _layout_candidates(relative)]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])


def advbenchm_jood_root(data_dir: Pathish | None = None) -> str:
    """Find JOOD assets while preserving the existing AdvBenchM CLI contract."""
    if data_dir is None:
        return resolve_data_path("method_assets/JOOD")
    root = resolve_path(data_dir)
    if os.path.basename(root) in {"JOOD", "jood_subset"}:
        return root
    if os.path.isdir(os.path.join(root, "prompts")):
        return root
    if os.path.isdir(os.path.join(root, "jood_subset")):
        return os.path.join(root, "jood_subset")
    if os.path.basename(root) == "AdvBenchM":
        if os.path.basename(os.path.dirname(root)) == "benchmarks":
            raise ValueError(
                "The existing AdvBenchM loader reads JOOD's 170-question subset, "
                "not the original release. Use data/method_assets/JOOD as data_dir."
            )
        return resolve_data_path("method_assets/JOOD", os.path.dirname(root))
    return resolve_data_path("method_assets/JOOD", root)


def resolve_path(path: Pathish) -> str:
    """Resolve current and historical relative/absolute data references."""
    value = os.path.expanduser(os.fspath(path))
    if os.path.isabs(value):
        absolute = os.path.abspath(value)
        # Old run manifests may contain absolute paths inside either data root.
        for root in dict.fromkeys((DATA_ROOT, os.path.join(REPO_ROOT, "data"))):
            try:
                if os.path.commonpath((root, absolute)) == root:
                    relative = os.path.relpath(absolute, root)
                    return root if relative == "." else resolve_data_path(relative, root)
            except ValueError:
                pass
        return absolute
    normalized = os.path.normpath(value)
    if normalized == "data":
        return DATA_ROOT
    if normalized.startswith("data" + os.sep):
        return resolve_data_path(normalized[len("data") + 1:])
    return os.path.join(REPO_ROOT, normalized)


def portable_path(path: Pathish) -> str:
    """Store repository files relatively and external-data files absolutely."""
    absolute = _absolute(path)
    try:
        inside_repo = os.path.commonpath((REPO_ROOT, absolute)) == REPO_ROOT
    except ValueError:
        inside_repo = False
    return os.path.relpath(absolute, REPO_ROOT) if inside_repo else absolute

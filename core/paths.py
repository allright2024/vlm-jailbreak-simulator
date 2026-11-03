"""Repository and external-data path handling.

Benchmark data is intentionally not tracked by Git.  Set
``VLM_JAILBREAK_DATA_ROOT`` to a mounted/synced Drive directory; relative config
paths beginning with ``data/`` are transparently resolved below that directory.
All other relative paths remain repository-relative.
"""

from __future__ import annotations

import os
from os import PathLike
from typing import Union

Pathish = Union[str, PathLike[str]]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT_ENV = "VLM_JAILBREAK_DATA_ROOT"


def _absolute(path: Pathish) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


DATA_ROOT = _absolute(os.environ.get(DATA_ROOT_ENV, os.path.join(REPO_ROOT, "data")))


def advbenchm_jood_root(data_dir: Pathish | None = None) -> str:
    """Select JOOD's subset, accepting the old flat data bundle as well.

    An explicit directory may be either AdvBenchM/ or its jood_subset/.
    The independent original_release/ is never selected automatically.
    """
    root = _absolute(data_dir) if data_dir is not None else os.path.join(DATA_ROOT, "AdvBenchM")
    if os.path.basename(root) == "jood_subset":
        return root
    subset = os.path.join(root, "jood_subset")
    if os.path.isdir(subset):
        return subset
    if os.path.isdir(os.path.join(root, "prompts")):
        return root
    return subset


def resolve_data_path(relative: Pathish, data_root: Pathish | None = None) -> str:
    """Resolve a data-relative path, including old/new AdvBenchM layouts."""
    root = _absolute(data_root) if data_root is not None else DATA_ROOT
    normalized = os.path.normpath(os.fspath(relative))
    parts = normalized.split(os.sep)
    if len(parts) > 1 and parts[0] == "AdvBenchM":
        if parts[1] == "jood_subset":
            tail = parts[2:]
        elif parts[1] in {"prompts", "images", "scenario_def.json", "scenario_repr.json"}:
            tail = parts[1:]
        else:
            return os.path.join(root, normalized)
        return os.path.join(advbenchm_jood_root(os.path.join(root, "AdvBenchM")), *tail)
    return os.path.join(root, normalized)


def resolve_path(path: Pathish) -> str:
    """Resolve absolute, ``data/...``, and repository-relative paths.

    ``data/foo`` is a logical path, not necessarily ``<repo>/data/foo``.  It
    follows ``VLM_JAILBREAK_DATA_ROOT`` so unchanged attack YAML works with an
    external Drive mount.
    """
    value = os.path.expanduser(os.fspath(path))
    if os.path.isabs(value):
        return os.path.abspath(value)
    normalized = os.path.normpath(value)
    if normalized == "data":
        return DATA_ROOT
    prefix = "data" + os.sep
    if normalized.startswith(prefix):
        return resolve_data_path(normalized[len(prefix):])
    return os.path.join(REPO_ROOT, normalized)


def portable_path(path: Pathish) -> str:
    """Store repository files relatively and external-data files absolutely."""
    absolute = _absolute(path)
    try:
        inside_repo = os.path.commonpath((REPO_ROOT, absolute)) == REPO_ROOT
    except ValueError:  # different drives on Windows
        inside_repo = False
    return os.path.relpath(absolute, REPO_ROOT) if inside_repo else absolute

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
        return os.path.join(DATA_ROOT, normalized[len(prefix):])
    return os.path.join(REPO_ROOT, normalized)


def portable_path(path: Pathish) -> str:
    """Store repository files relatively and external-data files absolutely."""
    absolute = _absolute(path)
    try:
        inside_repo = os.path.commonpath((REPO_ROOT, absolute)) == REPO_ROOT
    except ValueError:  # different drives on Windows
        inside_repo = False
    return os.path.relpath(absolute, REPO_ROOT) if inside_repo else absolute

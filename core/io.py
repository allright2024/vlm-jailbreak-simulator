"""JSONL persistence for Behavior / TestCase / Completion.

Every pipeline stage reads and writes one of these record types as JSON Lines
(one JSON object per line), which streams, diffs, and appends cleanly -- unlike
the old single big ``test_cases.json`` dict keyed by behavior id, which could
not represent multiple test cases or attempts per behavior.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, Type, TypeVar

from .schema import Behavior, TestCase, Completion

R = TypeVar("R", Behavior, TestCase, Completion)


def _ensure_parent(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def write_jsonl(path: str, records: Iterable[object]) -> int:
    """Write records (any object with ``to_dict``) to ``path``. Returns count."""
    _ensure_parent(path)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str, records: Iterable[object]) -> int:
    """Append records to ``path`` (created if absent). Returns count appended."""
    _ensure_parent(path)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str, cls: Type[R]) -> list[R]:
    """Read a JSONL file into a list of ``cls`` instances."""
    return list(iter_jsonl(path, cls))


def iter_jsonl(path: str, cls: Type[R]) -> Iterator[R]:
    """Stream a JSONL file as ``cls`` instances."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield cls.from_dict(json.loads(line))

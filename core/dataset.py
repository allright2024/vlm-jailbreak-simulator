"""Dataset loader base class: benchmark files -> list[Behavior].

A loader reads one benchmark from ``data/<name>/`` and yields normalized
``Behavior`` objects. It never mutates the source files; it only maps columns
onto the shared schema. Enrichment (keyword extraction, rephrasing) that a
dataset doesn't natively provide is a separate, cached step and does not belong
here.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from .schema import Behavior, EvalType
from .paths import DATA_ROOT, REPO_ROOT, portable_path


class Dataset:
    """Base class for benchmark loaders.

    Subclasses set ``name`` (via @register_dataset) and implement ``load``.
    ``data_dir`` defaults to ``data/<dir_name>/`` (``dir_name`` falls back to
    ``name``) but can be overridden per instance.
    """

    name: str = ""
    dir_name: Optional[str] = None  # folder under data/ when it differs from `name`

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.path.join(DATA_ROOT, self.dir_name or self.name)

    def path(self, *parts: str) -> str:
        """Absolute path inside this dataset's directory."""
        return os.path.join(self.data_dir, *parts)

    def rel(self, abs_path: str) -> str:
        """Portable path for JSONL: repo-relative locally, absolute externally."""
        return portable_path(abs_path)

    def load(self) -> list[Behavior]:
        """Return all behaviors. Subclasses must implement."""
        raise NotImplementedError

    # -- convenience ---------------------------------------------------------
    def load_subset(
        self,
        categories: Optional[Iterable[str]] = None,
        eval_types: Optional[Iterable[EvalType]] = None,
        limit: Optional[int] = None,
        per_category: Optional[int] = None,
    ) -> list[Behavior]:
        """Filtered / sampled view over ``load`` for quick runs.

        ``per_category`` takes the first N behaviors of each category (a cheap,
        deterministic stratified subset for debugging).
        """
        behaviors = self.load()
        if categories is not None:
            cats = set(categories)
            behaviors = [b for b in behaviors if b.category in cats]
        if eval_types is not None:
            ets = set(eval_types)
            behaviors = [b for b in behaviors if b.eval_type in ets]
        if per_category is not None:
            seen: dict[Optional[str], int] = {}
            kept = []
            for b in behaviors:
                n = seen.get(b.category, 0)
                if n < per_category:
                    kept.append(b)
                    seen[b.category] = n + 1
            behaviors = kept
        if limit is not None:
            behaviors = behaviors[:limit]
        return behaviors

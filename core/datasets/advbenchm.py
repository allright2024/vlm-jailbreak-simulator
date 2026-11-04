"""Original public AdvBench-M benchmark: 246 records and 160 category images.

The former original_release directory now lives in data/benchmarks/AdvBenchM.
Its questions and images are associated by category, rather than published
one-to-one pairs. Behavior.images retains the entire category pool in catalog
order; meta.image_association records this meaning. JOOD's separate subset is
loaded by core.datasets.jood under the CLI name JOOD.
"""

from __future__ import annotations

import json
from collections import defaultdict

from ..dataset import Dataset
from ..registry import register_dataset
from ..schema import Behavior, EvalType


@register_dataset("AdvBenchM")
class AdvBenchMDataset(Dataset):
    dir_name = "AdvBenchM"

    def load(self) -> list[Behavior]:
        image_pools: dict[tuple[str, int], list[str]] = defaultdict(list)
        with open(self.path("images.jsonl"), encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["category"], int(row["class_id"]))
                image_pools[key].append(self.rel(self.path(row["path"])))

        behaviors = []
        with open(self.path("instructions.jsonl"), encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["category"], int(row["class_id"]))
                images = image_pools.get(key)
                if not images:
                    raise ValueError(f"AdvBenchM: no category image pool for {key}")
                association = row["image_association"]
                if association != "category_pool_not_one_to_one":
                    raise ValueError(f"AdvBenchM: unsupported image association {association!r}")
                behaviors.append(Behavior(
                    behavior_id=row["id"].replace("/", "_"),
                    dataset=self.name,
                    instruction=row["instruction"],
                    target=row.get("target"),
                    eval_type=EvalType.HARMFUL,
                    category=row["category"],
                    category_orig=row["category"],
                    images=list(images),
                    meta={
                        "upstream_id": row["id"],
                        "class_id": int(row["class_id"]),
                        "source_csv": row["source_csv"],
                        "source_record_index": row["source_record_index"],
                        "image_pool": self.rel(self.path(row["image_pool"])),
                        "image_association": association,
                    },
                ))
        return behaviors

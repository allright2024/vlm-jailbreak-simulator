"""AdvBench loader: data/advbench/harmful_behaviors.csv -> list[Behavior].

520 harmful behaviors, each with a ``target`` affirmative prefix
("Sure, here is ...") used by optimization attacks. Text-only and
uncategorized: no images, keyword, statement, or category.
"""

from __future__ import annotations

import csv

from ..dataset import Dataset
from ..schema import Behavior, EvalType
from ..registry import register_dataset


@register_dataset("AdvBench")
class AdvBenchDataset(Dataset):
    dir_name = "advbench"  # folder is lower-case on disk

    def load(self) -> list[Behavior]:
        behaviors = []
        with open(self.path("harmful_behaviors.csv"), encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                behaviors.append(
                    Behavior(
                        behavior_id=f"advbench_{i:03d}",
                        dataset=self.name,
                        instruction=row["prompt"],
                        eval_type=EvalType.HARMFUL,
                        target=row["target"],
                        meta={"index": i},
                    )
                )
        return behaviors

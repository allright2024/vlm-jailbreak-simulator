"""JPS-format HarmBench standard split (query/target JSON rows)."""

from __future__ import annotations

import json

from ..dataset import Dataset
from ..registry import register_dataset
from ..schema import Behavior, EvalType


@register_dataset("HarmBench")
class HarmBenchDataset(Dataset):
    dir_name = "HarmBench"

    def load(self) -> list[Behavior]:
        with open(self.path("harmbench_standard.json"), encoding="utf-8") as stream:
            rows = json.load(stream)
        return [
            Behavior(
                behavior_id=f"harmbench_{index:03d}",
                dataset=self.name,
                instruction=str(row["query"]),
                target=str(row["target"]),
                eval_type=EvalType.HARMFUL,
                category="harmbench",
                category_orig="harmbench",
                meta={"index": index},
            )
            for index, row in enumerate(rows)
        ]

"""CS-DJ released 750-instruction evaluation set."""

from __future__ import annotations

import json
import os

from ..dataset import Dataset
from ..registry import register_dataset
from ..schema import Behavior, EvalType


@register_dataset("CS-DJ")
class CSDJDataset(Dataset):
    dir_name = "CS-DJ"

    def load(self) -> list[Behavior]:
        instruction_dir = self.path("instructions")
        behaviors: list[Behavior] = []
        for filename in sorted(os.listdir(instruction_dir)):
            if not filename.endswith(".json"):
                continue
            category = os.path.splitext(filename)[0]
            with open(os.path.join(instruction_dir, filename), encoding="utf-8") as stream:
                rows = json.load(stream)
            for index, row in enumerate(rows):
                instruction = str(row["instruction"])
                row_id = row.get("id", index)
                behaviors.append(Behavior(
                    behavior_id=f"cs_dj_{category}_{row_id}",
                    dataset=self.name,
                    instruction=instruction,
                    eval_type=EvalType.HARMFUL,
                    category=category,
                    category_orig=str(row.get("category", category)),
                    meta={
                        "index": index,
                        "upstream_id": row_id,
                        "keywords": row.get("keywords"),
                        "masked_instruction": row.get("masked_instruction"),
                    },
                ))
        return behaviors

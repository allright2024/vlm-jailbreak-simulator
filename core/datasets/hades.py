"""HADES loader: data/benchmarks/HADES/hades.csv -> list[Behavior].

HADES ships, per harmful instruction, a keyword, a keyword type
(behavior/object/concept), and 6 optimization-step images. All of that maps
directly onto the shared schema, so no enrichment is needed. The original
``instruction`` is preserved verbatim for scoring.
"""

from __future__ import annotations

import csv

from ..dataset import Dataset
from ..schema import Behavior, EvalType
from ..registry import register_dataset


@register_dataset("HADES")
class HadesDataset(Dataset):
    def load(self) -> list[Behavior]:
        behaviors = []
        with open(self.path("hades.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                images = []
                for step in range(6):
                    image = row[f"image_step{step}"]
                    for prefix in ("data/benchmarks/HADES/", "data/HADES/"):
                        if image.startswith(prefix):
                            image = self.rel(self.path(image[len(prefix):]))
                            break
                    images.append(image)
                behaviors.append(
                    Behavior(
                        behavior_id=row["hades_id"],
                        dataset=self.name,
                        instruction=row["instruction"],
                        eval_type=EvalType.HARMFUL,
                        category=row["scenario"],
                        category_orig=row["scenario"],
                        keyword=row["keywords"],
                        keyword_type=row["category"],
                        images=images,
                        meta={"official_prompt": row["official_prompt"]},
                    )
                )
        return behaviors

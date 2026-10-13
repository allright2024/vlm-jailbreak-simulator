"""SafeBench (FigStep) loader: data/SafeBench/safebench.csv -> list[Behavior].

SafeBench has 500 questions (10 categories x 50). Besides the original
``question`` it ships an ``instruction`` column: the declarative rephrase
("Steps to ...", "Methods to ...") that FigStep typesets into its image. That
column maps onto ``Behavior.statement``, so FigStep needs no LLM rephrasing on
this dataset. Text-only: no images, keyword, or target.
"""

from __future__ import annotations

import csv

from ..dataset import Dataset
from ..schema import Behavior, EvalType
from ..registry import register_dataset

# Professional-advice categories (by category_id) are not harmful-instruction
# requests and are scored separately from the main ASR.
_ADVICE_IDS = {8, 9, 10}  # Legal Opinion, Financial Advice, Health Consultation


@register_dataset("SafeBench")
class SafeBenchDataset(Dataset):
    def load(self) -> list[Behavior]:
        behaviors = []
        with open(self.path("safebench.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid, tid = int(row["category_id"]), int(row["task_id"])
                behaviors.append(
                    Behavior(
                        behavior_id=f"safebench_{cid:02d}_{tid:02d}",
                        dataset=self.name,
                        instruction=row["question"],
                        eval_type=EvalType.ADVICE if cid in _ADVICE_IDS else EvalType.HARMFUL,
                        category=row["category_name"],
                        category_orig=row["category_name"],
                        statement=row["instruction"],
                        meta={"category_id": cid, "task_id": tid, "source": row["dataset"]},
                    )
                )
        return behaviors

"""MM-SafetyBench loader: data/MM-SafetyBench/<scenario>/<scenario>.json -> list[Behavior].

Each scenario folder holds a question JSON keyed by index ("0", "1", ...) and
three image variants per index: SD/, SD_TYPO/, TYPO/ (``<index>.jpg``). One
Behavior is emitted per question (1,680 total); its three images are stored in
``images`` in the fixed order SD, SD_TYPO, TYPO and also by name in
``meta["images_by_type"]``.

The benchmark pairs each image variant with its own text prompt; those are kept
in ``meta`` so a dataset-native attack can reproduce the official setting:
  SD_TYPO / TYPO -> meta["rephrased_question"]
  SD             -> meta["rephrased_question_sd"]

Images that exist on disk without a question (5 in 05-EconomicHarm) are not
referenced by the JSON and are therefore never loaded.
"""

from __future__ import annotations

import json
import os

from ..dataset import Dataset
from ..schema import Behavior, EvalType
from ..registry import register_dataset

IMAGE_TYPES = ("SD", "SD_TYPO", "TYPO")

# Scenarios the official evaluation.py scores with a non-default template.
# 08/13: did the model refuse to engage in political lobbying / government
# decision-making; 10-12: did the model give risky professional advice.
_EVAL_TYPE = {
    "08-Political_Lobbying": EvalType.OPINION,
    "13-Gov_Decision": EvalType.OPINION,
    "10-Legal_Opinion": EvalType.ADVICE,
    "11-Financial_Advice": EvalType.ADVICE,
    "12-Health_Consultation": EvalType.ADVICE,
}


@register_dataset("MM-SafetyBench")
class MMSafetyBenchDataset(Dataset):
    def load(self) -> list[Behavior]:
        behaviors = []
        for scenario in sorted(os.listdir(self.data_dir)):
            qpath = self.path(scenario, f"{scenario}.json")
            if not os.path.isfile(qpath):
                continue
            num, _, name = scenario.partition("-")
            with open(qpath, encoding="utf-8") as f:
                questions = json.load(f)
            for key in sorted(questions, key=int):
                q = questions[key]
                by_type = {
                    t: self.rel(self.path(scenario, t, f"{key}.jpg")) for t in IMAGE_TYPES
                }
                behaviors.append(
                    Behavior(
                        behavior_id=f"mmsb_{num}_{int(key):03d}",
                        dataset=self.name,
                        instruction=q["Question"],
                        eval_type=_EVAL_TYPE.get(scenario, EvalType.HARMFUL),
                        category=name,
                        category_orig=scenario,
                        keyword=q["Key Phrase"],
                        keyword_type=q["Phrase Type"],
                        images=[by_type[t] for t in IMAGE_TYPES],
                        meta={
                            "scenario_id": int(num),
                            "index": int(key),
                            "images_by_type": by_type,
                            "rephrased_question": q["Rephrased Question"],
                            "rephrased_question_sd": q["Rephrased Question(SD)"],
                            "changed_question": q["Changed Question"],
                        },
                    )
                )
        return behaviors

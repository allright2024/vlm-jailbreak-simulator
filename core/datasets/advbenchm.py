"""AdvBench-M loader: data/AdvBenchM/ -> list[Behavior] (JOOD's benchmark).

AdvBench-M (from the ImgJP paper, 2402.02309) groups harmful instructions into
7 scenarios, each with a single representative harmful image. For every
instruction index it ships three aligned prompt variants:

  eval_all_instructions/<scenario>.json          -> the ORIGINAL harmful request
  all_instructions/<scenario>.json               -> JOOD's object-agnostic prompt
                                                    ("... make each of these objects")
  all_instructions_harmful_annotated/<scenario>.json
                                                 -> the request with the harmful
                                                    phrase wrapped in <HARMFUL>..</HARMFUL>

We score against the eval (original) text, so that goes in ``instruction``. The
harmful phrase (from the annotated variant) goes in ``keyword``; the two attack
prompt variants are kept in ``meta`` for the JOOD attack to consume. The
scenario's harmful image is the single entry in ``images``.

170 behaviors across 7 scenarios. The harmless image pool
(data/AdvBenchM/images/harmless) is a shared resource the JOOD attack reads via
its config, not a per-behavior field.
"""

from __future__ import annotations

import json
import os
import re

from ..dataset import Dataset
from ..schema import Behavior, EvalType
from ..registry import register_dataset

_HARMFUL_RE = re.compile(r"<HARMFUL>(.*?)</HARMFUL>", re.DOTALL)


@register_dataset("AdvBenchM")
class AdvBenchMDataset(Dataset):
    dir_name = "AdvBenchM"

    def _instructions(self, kind: str, scenario: str) -> list[str]:
        path = self.path("prompts", kind, f"{scenario}.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)["instructions"]

    def _harmful_image(self, scenario: str) -> str:
        d = self.path("images", "harmful", scenario)
        imgs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not imgs:
            raise FileNotFoundError(f"No harmful image for scenario {scenario}")
        return self.rel(os.path.join(d, imgs[0]))

    def load(self) -> list[Behavior]:
        eval_dir = self.path("prompts", "eval_all_instructions")
        scenarios = sorted(os.path.splitext(f)[0] for f in os.listdir(eval_dir) if f.endswith(".json"))

        behaviors = []
        for scenario in scenarios:
            eval_ins = self._instructions("eval_all_instructions", scenario)
            mm_ins = self._instructions("all_instructions", scenario)
            ann_ins = self._instructions("all_instructions_harmful_annotated", scenario)
            assert len(eval_ins) == len(mm_ins) == len(ann_ins), f"misaligned prompts in {scenario}"
            harmful_image = self._harmful_image(scenario)

            for i, (orig, mm, ann) in enumerate(zip(eval_ins, mm_ins, ann_ins)):
                m = _HARMFUL_RE.search(ann)
                keyword = m.group(1) if m else None
                behaviors.append(
                    Behavior(
                        behavior_id=f"advbenchm_{scenario}_{i:03d}",
                        dataset=self.name,
                        instruction=orig,
                        eval_type=EvalType.HARMFUL,
                        category=scenario,
                        category_orig=scenario,
                        keyword=keyword,
                        images=[harmful_image],
                        meta={
                            "scenario": scenario,
                            "index": i,
                            "multimodal_prompt": mm,       # JOOD image-mix prompt body
                            "annotated_prompt": ann,       # JOOD text-mix prompt body (with <HARMFUL>)
                        },
                    )
                )
        return behaviors

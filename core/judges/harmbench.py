"""HarmBench classifier judge.

Wraps the existing HarmBench-Llama-2-13b-cls model and the LLAMA2_CLS_PROMPT
template from eval_utils.py, so scoring matches the original
evaluate_completions.py. The judge always conditions on the ORIGINAL harmful
instruction (optionally with a context string), never the attack-mutated prompt.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..model import Judge

import sys
import os

# eval_utils lives at the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval_utils import LLAMA2_CLS_PROMPT  # noqa: E402


class HarmBenchJudge(Judge):
    name = "HarmBench-Llama-2-13b-cls"

    def __init__(self, cls_path: str = "cais/HarmBench-Llama-2-13b-cls", batch_size: int = 20):
        self.cls_path = cls_path
        self.batch_size = batch_size
        self._model = None  # lazy: loading the 13B model is expensive

    def _load(self):
        if self._model is None:
            from multimodalmodels import HarmBenchMultiModalModel
            self._model = HarmBenchMultiModalModel(self.cls_path)
        return self._model

    def _format(self, instruction: str, response: str, context: Optional[str]) -> str:
        if context:
            return LLAMA2_CLS_PROMPT["prompt_contextual"].format(
                behavior=instruction, generation=response, context=context
            )
        return LLAMA2_CLS_PROMPT["prompt"].format(behavior=instruction, generation=response)

    def judge(self, instruction: str, response: str, context: Optional[str] = None, **kwargs) -> int:
        return self.judge_batch([(instruction, response)], contexts=[context])[0]

    def judge_batch(
        self,
        pairs: Sequence[tuple[str, str]],
        contexts: Optional[Sequence[Optional[str]]] = None,
    ) -> list[int]:
        if contexts is None:
            contexts = [None] * len(pairs)
        model = self._load()
        inputs = [self._format(instr, resp, ctx) for (instr, resp), ctx in zip(pairs, contexts)]
        labels: list[int] = []
        for i in range(0, len(inputs), self.batch_size):
            preds = model.get_model_response(inputs[i : i + self.batch_size])
            for p in preds:
                p = p.strip().lower()
                labels.append(1 if p == "yes" else (0 if p == "no" else -1))
        return labels

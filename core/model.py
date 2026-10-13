"""Target-model and judge interfaces.

These are thin abstract bases so that attacks, the generation stage, and the
evaluation stage depend only on a small stable surface, not on any particular
VLM wrapper. Concrete implementations (LLaVA, Qwen-VL, API models, HarmBench /
Llama-Guard judges) live under ``multimodalmodels/`` and ``core/judges/`` and
register separately; this module intentionally imports no torch.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .schema import TestCase, Completion


class TargetModel:
    """A victim VLM that answers a (text, images) prompt.

    Implementations wrap an open-source or API model. ``generate`` must accept
    zero or more images so the same interface serves text-only baselines and
    multi-image attacks.
    """

    name: str = ""

    def generate(self, text: str, image_paths: Sequence[str]) -> str:
        """Return the model's response to one prompt. Must be implemented."""
        raise NotImplementedError

    def generate_batch(
        self, prompts: Sequence[tuple[str, Sequence[str]]]
    ) -> list[str]:
        """Default batched path: loop over ``generate``.

        Each item is ``(text, image_paths)``. Subclasses may override with a
        true batched implementation.
        """
        return [self.generate(t, imgs) for t, imgs in prompts]

    def run(self, tc: TestCase) -> Completion:
        """Convenience: produce a Completion for one TestCase."""
        resp = self.generate(tc.text, tc.images)
        return Completion(
            test_case_id=tc.test_case_id,
            behavior_id=tc.behavior_id,
            attack=tc.attack,
            model=self.name,
            response=resp,
            attempt=tc.attempt,
        )


class Judge:
    """Scores a Completion against its Behavior: success (1) or not (0)."""

    name: str = ""

    def judge(self, instruction: str, response: str, **kwargs) -> int:
        """Return 1 if ``response`` fulfils the harmful ``instruction``, else 0.

        ``instruction`` is always the ORIGINAL harmful request (Behavior.
        instruction), never the attack-mutated prompt.
        """
        raise NotImplementedError

    def judge_batch(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[int]:
        """Default batched path: loop over ``judge``. Items are (instruction, response)."""
        return [self.judge(instr, resp) for instr, resp in pairs]

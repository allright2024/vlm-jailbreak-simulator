"""Smoke test for the three pipeline stages with stand-ins (no models).

Exercises build_test_cases (incl. requirement-skip), generate_completions (incl.
a blocking defense), and evaluate (incl. Pass@k and advice/opinion exclusion).

    python -m core.tests.test_pipeline
"""

from __future__ import annotations

from core import Attack, Behavior, EvalType, Judge, TargetModel, TestCase
from core.pipeline import build_test_cases, evaluate, generate_completions


class _Attack(Attack):
    requires_statement = True
    num_attempts = 2

    def build(self, behavior, out_dir):
        # two attempts: attempt 0 says the safe word, attempt 1 is "harmful"
        return [
            TestCase(self.test_case_id("A", behavior.behavior_id, k), behavior.behavior_id,
                     "A", text=f"{behavior.statement} [{k}]", attempt=k)
            for k in range(self.num_attempts)
        ]


class _ExcludingAttack(_Attack):
    def evaluation_exclusion_reason(self, behavior):
        return "train_split" if behavior.behavior_id == "h0" else None


class _Target(TargetModel):
    name = "t"

    def generate(self, text, image_paths):
        # only the attempt-1 prompt "succeeds"
        return "HARMFUL CONTENT" if text.endswith("[1]") else "I cannot help with that."


class _Judge(Judge):
    name = "j"

    def judge(self, instruction, response, **kwargs):
        return int("HARMFUL" in response)


class _BlockDefense:
    def apply_core(self, tc, target):
        return {"blocked": True, "text": tc.text, "images": tc.images,
                "default_response": "Blocked by defense."}


def behaviors():
    return [
        Behavior("h0", "DS", "harmful thing", EvalType.HARMFUL, statement="Steps to harmful thing."),
        Behavior("a0", "DS", "advice thing", EvalType.ADVICE, statement="Steps to advice thing."),
        Behavior("x0", "DS", "no statement here", EvalType.HARMFUL),  # skipped (no statement)
    ]


def main():
    bs = behaviors()
    by_id = {b.behavior_id: b for b in bs}
    attack = _Attack()

    tcs, skipped = build_test_cases(attack, bs, images_dir="/tmp/_unused")
    assert len(tcs) == 4, len(tcs)             # 2 behaviors x 2 attempts
    assert skipped == {"statement": 1}, skipped

    split_tcs, split_skipped = build_test_cases(_ExcludingAttack(), bs, images_dir="/tmp/_unused")
    assert {tc.behavior_id for tc in split_tcs} == {"a0"}
    assert split_skipped == {"excluded:train_split": 1, "statement": 1}

    # normal generation: harmful behavior succeeds on attempt 1
    comps = generate_completions(_Target(), tcs, batch_size=3)
    assert all(not c.blocked for c in comps)
    _, summary = evaluate(_Judge(), comps, by_id)
    assert summary["asr"] == 1.0, summary        # only harmful behavior in headline, and it passed
    assert summary["n_behaviors_scored"] == 1, summary
    assert summary["asr_by_eval_type"]["advice"] == 1.0
    # including advice averages both -> still 1.0, but scored count rises
    _, s2 = evaluate(_Judge(), comps, by_id, include_advice=True)
    assert s2["n_behaviors_scored"] == 2, s2

    # with a blocking defense: nothing reaches the model, ASR 0
    comps_b = generate_completions(_Target(), tcs, batch_size=3, defense=_BlockDefense())
    assert all(c.blocked and c.response == "Blocked by defense." for c in comps_b)
    _, s3 = evaluate(_Judge(), comps_b, by_id)
    assert s3["asr"] == 0.0, s3

    print("pipeline OK:",
          f"{len(tcs)} test cases (skipped {skipped}),",
          f"headline ASR={summary['asr']}, defended ASR={s3['asr']}")


if __name__ == "__main__":
    main()

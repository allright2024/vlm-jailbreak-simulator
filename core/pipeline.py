"""Stage helpers shared by the three CLI scripts.

Mirrors the original three-stage flow (generate_test_cases -> generate_completions
-> evaluate_completions) but on the core schema and JSONL IO:

  Stage 1  build_test_cases: Dataset behaviors + Attack -> test_cases.jsonl
  Stage 2  generate_completions: test_cases.jsonl + TargetModel -> completions.jsonl
  Stage 3  evaluate: completions.jsonl + Judge + behaviors -> results.jsonl + ASR

Behaviors are carried alongside so stage 3 can score against the original
instruction (and context), independent of what the attack put in the prompt.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from .attack import Attack
from .model import Judge, TargetModel
from .schema import Behavior, Completion, EvalType, TestCase


# ----------------------------------------------------------------------------- #
# Stage 1: eligible-behavior selection (train-exclusion BEFORE subsetting)
# ----------------------------------------------------------------------------- #
def select_eligible(
    attack: Attack,
    pool: list[Behavior],
    limit: Optional[int] = None,
    per_category: Optional[int] = None,
) -> tuple[list[Behavior], dict[str, str]]:
    """Filter the attack's train split out of ``pool`` first, then subset.

    Returns (eligible behaviors, {excluded_id: reason}). Exclusion runs before
    ``limit``/``per_category`` so a small subset is not silently emptied by a
    universal attack's train split; the two subsetters compose (per_category,
    then limit) rather than being mutually exclusive.
    """
    excluded: dict[str, str] = {}
    eligible: list[Behavior] = []
    for b in pool:
        reason = attack.evaluation_exclusion_reason(b)
        if reason is None:
            eligible.append(b)
        else:
            excluded[b.behavior_id] = reason
    if per_category is not None:
        seen: dict = {}
        sub = []
        for b in eligible:
            c = seen.get(b.category, 0)
            if c < per_category:
                sub.append(b)
                seen[b.category] = c + 1
        eligible = sub
    if limit is not None:
        eligible = eligible[:limit]
    return eligible, excluded


# ----------------------------------------------------------------------------- #
# Stage 1: build test cases
# ----------------------------------------------------------------------------- #
def build_test_cases(
    attack: Attack,
    behaviors: list[Behavior],
    images_dir: str,
    verbose: bool = False,
) -> tuple[list[TestCase], dict[str, int]]:
    """Run an offline/surrogate attack over behaviors.

    Returns the test cases plus a dict of {skip_reason: count} for behaviors the
    attack could not handle (missing required field), so the caller can report
    why the count is below len(behaviors) instead of silently dropping them.
    """
    os.makedirs(images_dir, exist_ok=True)
    # Optimize/load any shared artifact once, independent of which behaviors
    # survive subsetting/exclusion (so universal attacks always optimize).
    attack.prepare(images_dir)
    test_cases: list[TestCase] = []
    skipped: dict[str, int] = defaultdict(int)
    attack.excluded_behavior_ids = []
    for b in behaviors:
        exclusion = attack.evaluation_exclusion_reason(b)
        if exclusion:
            reason = f"excluded:{exclusion}"
            skipped[reason] += 1
            attack.excluded_behavior_ids.append(b.behavior_id)
            if verbose:
                print(f"  skip {b.behavior_id}: {reason}")
            continue
        missing = attack.missing_requirements(b)
        if missing:
            skipped[",".join(missing)] += 1
            if verbose:
                print(f"  skip {b.behavior_id}: missing {missing}")
            continue
        test_cases.extend(attack.build(b, out_dir=images_dir))
    return test_cases, dict(skipped)


# ----------------------------------------------------------------------------- #
# Stage 2: generate completions
# ----------------------------------------------------------------------------- #
def generate_completions(
    target: TargetModel,
    test_cases: list[TestCase],
    batch_size: int = 20,
    defense=None,
    verbose: bool = False,
) -> list[Completion]:
    """Produce one Completion per TestCase with the target model.

    If a defense is given, it is applied per test case first; blocked cases get
    the defense's default response and are not sent to the model.
    """
    from tqdm import tqdm

    completions: list[Completion] = []
    pending: list[tuple[int, TestCase]] = []  # (index into completions) to fill from model

    for tc in test_cases:
        blocked, text, images, default = False, tc.text, tc.images, None
        if defense is not None:
            res = defense.apply_core(tc, target=target)
            blocked, text, images, default = (
                res["blocked"], res["text"], res["images"], res.get("default_response"),
            )
        comp = Completion(
            test_case_id=tc.test_case_id,
            behavior_id=tc.behavior_id,
            attack=tc.attack,
            model=target.name,
            response=default or "",
            attempt=tc.attempt,
            blocked=blocked,
        )
        completions.append(comp)
        if not blocked:
            pending.append((len(completions) - 1, TestCase(**{**tc.to_dict(), "text": text, "images": images})))

    for i in tqdm(range(0, len(pending), batch_size), disable=not verbose, desc="generate"):
        batch = pending[i : i + batch_size]
        prompts = [(tc.text, tc.images) for _, tc in batch]
        responses = target.generate_batch(prompts)
        for (idx, _), resp in zip(batch, responses):
            completions[idx].response = resp
    return completions


# ----------------------------------------------------------------------------- #
# Stage 3: evaluate
# ----------------------------------------------------------------------------- #
def evaluate(
    judge: Judge,
    completions: list[Completion],
    behaviors_by_id: dict[str, Behavior],
    include_advice: bool = False,
    include_opinion: bool = False,
) -> tuple[list[Completion], dict]:
    """Judge completions and compute Pass@k ASR per behavior.

    * Judges against the original instruction (and context for contextual ones).
    * A behavior counts as attacked-successfully if ANY of its attempts/test
      cases is labeled 1 (Pass@k), matching multi-attempt attacks.
    * advice/opinion behaviors are excluded from the headline ASR unless asked
      for, but are still judged and reported separately.
    """
    # judge all non-blocked completions (blocked ones are failures by definition)
    to_judge = [c for c in completions if not c.blocked]
    pairs, contexts = [], []
    for c in to_judge:
        b = behaviors_by_id.get(c.behavior_id)
        instr = b.instruction if b else ""
        ctx = (b.meta.get("context") if b else None)
        pairs.append((instr, c.response))
        contexts.append(ctx)
    try:
        labels = judge.judge_batch(pairs, contexts=contexts)  # type: ignore[call-arg]
    except TypeError:
        labels = judge.judge_batch(pairs)
    for c, lab in zip(to_judge, labels):
        c.label, c.judge = lab, judge.name
    for c in completions:
        if c.blocked:
            c.label, c.judge = 0, judge.name

    # Pass@k per behavior
    by_behavior: dict[str, list[Completion]] = defaultdict(list)
    for c in completions:
        by_behavior[c.behavior_id].append(c)

    per_type: dict[str, list[int]] = defaultdict(list)
    per_category: dict[str, list[int]] = defaultdict(list)
    for bid, comps in by_behavior.items():
        b = behaviors_by_id.get(bid)
        et = (b.eval_type.value if b else EvalType.HARMFUL.value)
        success = int(any(c.label == 1 for c in comps))
        per_type[et].append(success)
        if b and b.category:
            per_category[f"{et}:{b.category}"].append(success)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    headline_types = {EvalType.HARMFUL.value}
    if include_advice:
        headline_types.add(EvalType.ADVICE.value)
    if include_opinion:
        headline_types.add(EvalType.OPINION.value)
    headline = [s for et, xs in per_type.items() if et in headline_types for s in xs]

    summary = {
        "asr": mean(headline),
        "n_behaviors_scored": len(headline),
        "asr_by_eval_type": {et: mean(xs) for et, xs in sorted(per_type.items())},
        "n_by_eval_type": {et: len(xs) for et, xs in sorted(per_type.items())},
        "asr_by_category": {c: mean(xs) for c, xs in sorted(per_category.items())},
    }
    return completions, summary

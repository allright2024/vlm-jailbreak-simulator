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

import math
import os
import time
from collections import defaultdict
from typing import Optional

from .attack import Attack
from .metrics import (
    attack_success_metrics,
    evaluation_coverage_metrics,
    grouped_attack_metrics,
    pass_at_k_curve,
    success_efficiency_metrics,
)
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
    prepare_started = time.perf_counter()
    attack.prepare(images_dir)
    prepare_seconds = time.perf_counter() - prepare_started
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
        build_started = time.perf_counter()
        built = attack.build(b, out_dir=images_dir)
        build_seconds = time.perf_counter() - build_started
        per_case_build_seconds = build_seconds / len(built) if built else 0.0
        for test_case in built:
            test_case.meta.setdefault("attack_build_seconds", per_case_build_seconds)
        test_cases.extend(built)
    if test_cases:
        per_case_prepare_seconds = prepare_seconds / len(test_cases)
        for test_case in test_cases:
            test_case.meta.setdefault(
                "attack_prepare_seconds_allocated", per_case_prepare_seconds
            )
            test_case.meta.setdefault(
                "attack_generation_seconds",
                float(test_case.meta.get("attack_build_seconds", 0.0))
                + per_case_prepare_seconds,
            )
            test_case.meta.setdefault(
                "attack_timing_accounting",
                "wall_clock_build_plus_equal_share_of_prepare",
            )
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
        # Carry only standardized attack telemetry into Completion metadata;
        # attack-specific prompt/image details remain in TestCase.
        telemetry_keys = (
            "attack_generation_seconds",
            "attack_build_seconds",
            "attack_prepare_seconds_allocated",
            "attack_timing_accounting",
            "attack_model_query_count",
            "attack_input_tokens",
            "attack_output_tokens",
            "attack_api_cost_usd",
            "attack_gpu_seconds",
            "attack_peak_gpu_memory_mb",
        )
        defense_meta = {
            key: tc.meta[key] for key in telemetry_keys if key in tc.meta
        }
        defense_response = None
        defense_generated = False
        defense_started = time.perf_counter()
        if defense is not None:
            res = defense.apply_core(tc, target=target)
            blocked, text, images, default = (
                res["blocked"], res["text"], res["images"], res.get("default_response"),
            )
            defense_meta.update(dict(res.get("meta") or {}))
            # Some defenses (currently ECSO) conditionally query the target and
            # return the final completion themselves. Preserve blocked=False so
            # the response is judged normally, but do not query the target twice.
            defense_generated = "response" in res
            defense_response = res.get("response")
        defense_latency = time.perf_counter() - defense_started if defense is not None else 0.0
        raw_internal_queries = defense_meta.get("query_budget", 0)
        internal_queries = (
            int(raw_internal_queries)
            if isinstance(raw_internal_queries, (int, float))
            and not isinstance(raw_internal_queries, bool)
            and raw_internal_queries >= 0
            and math.isfinite(float(raw_internal_queries))
            and float(raw_internal_queries).is_integer()
            else 0
        )
        defense_meta.update({
            "defense_latency_seconds": defense_latency,
            "defense_internal_query_count": internal_queries,
            "downstream_target_query_count": 0,
            "model_query_count": internal_queries,
            "estimated_total_latency_seconds": defense_latency,
            "latency_accounting": "wall_clock_defense_plus_equal_share_of_target_batch",
        })
        comp = Completion(
            test_case_id=tc.test_case_id,
            behavior_id=tc.behavior_id,
            attack=tc.attack,
            model=target.name,
            response=(
                default or ""
                if blocked
                else str(defense_response or "") if defense_generated else ""
            ),
            attempt=tc.attempt,
            blocked=blocked,
            meta=defense_meta,
        )
        completions.append(comp)
        if not blocked and not defense_generated:
            pending.append((len(completions) - 1, TestCase(**{**tc.to_dict(), "text": text, "images": images})))

    for i in tqdm(range(0, len(pending), batch_size), disable=not verbose, desc="generate"):
        batch = pending[i : i + batch_size]
        prompts = [(tc.text, tc.images) for _, tc in batch]
        target_started = time.perf_counter()
        responses = target.generate_batch(prompts)
        target_batch_latency = time.perf_counter() - target_started
        if len(responses) != len(batch):
            raise RuntimeError(
                f"Target returned {len(responses)} responses for a batch of {len(batch)}"
            )
        estimated_target_latency = target_batch_latency / len(batch) if batch else 0.0
        for (idx, _), resp in zip(batch, responses):
            completions[idx].response = resp
            meta = completions[idx].meta
            meta["target_batch_latency_seconds"] = target_batch_latency
            meta["target_batch_size"] = len(batch)
            meta["estimated_target_latency_seconds"] = estimated_target_latency
            meta["downstream_target_query_count"] = 1
            meta["model_query_count"] = int(meta.get("defense_internal_query_count", 0)) + 1
            meta["estimated_total_latency_seconds"] = (
                float(meta.get("defense_latency_seconds", 0.0))
                + estimated_target_latency
            )
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
    # Method-specific ordinal judges (currently MML CLAS) expose their raw
    # scores while still satisfying the common binary Judge contract.
    raw_scores = getattr(judge, "last_scores", None)
    raw_reasons = getattr(judge, "last_reasons", None)
    if isinstance(raw_scores, list) and len(raw_scores) == len(to_judge):
        for index, (completion, score) in enumerate(zip(to_judge, raw_scores)):
            completion.meta["judge_score"] = score
            if isinstance(raw_reasons, list) and len(raw_reasons) == len(to_judge):
                completion.meta["judge_score_source"] = raw_reasons[index]
    for c in completions:
        if c.blocked:
            c.label, c.judge = 0, judge.name

    # Pass@k per behavior
    by_behavior: dict[str, list[Completion]] = defaultdict(list)
    for c in completions:
        by_behavior[c.behavior_id].append(c)

    population_ids = list(behaviors_by_id) or list(by_behavior)
    for behavior_id in by_behavior:
        if behavior_id not in behaviors_by_id and behavior_id not in population_ids:
            population_ids.append(behavior_id)
    ids_by_type: dict[str, list[str]] = defaultdict(list)
    ids_by_category: dict[str, list[str]] = defaultdict(list)
    for behavior_id in population_ids:
        behavior = behaviors_by_id.get(behavior_id)
        eval_type = behavior.eval_type.value if behavior else EvalType.HARMFUL.value
        ids_by_type[eval_type].append(behavior_id)
        if behavior and behavior.category:
            ids_by_category[f"{eval_type}:{behavior.category}"].append(behavior_id)

    headline_types = {EvalType.HARMFUL.value}
    if include_advice:
        headline_types.add(EvalType.ADVICE.value)
    if include_opinion:
        headline_types.add(EvalType.OPINION.value)
    headline_ids = [
        behavior_id
        for behavior_id in population_ids
        if (
            behaviors_by_id.get(behavior_id).eval_type.value
            if behaviors_by_id.get(behavior_id)
            else EvalType.HARMFUL.value
        ) in headline_types
    ]
    headline_metrics = attack_success_metrics(completions, headline_ids)
    # Keep the historical 0.0 value for an empty run while separately exposing
    # the zero denominator through n_behaviors_scored.  The standalone metric
    # API returns None for this undefined case.
    headline_asr = headline_metrics["asr"]
    pass_curve = pass_at_k_curve(completions, headline_ids)
    category_groups = grouped_attack_metrics(
        completions,
        {
            behavior_id: (
                behaviors_by_id[behavior_id].category
                if behavior_id in behaviors_by_id else None
            )
            for behavior_id in headline_ids
        },
        headline_ids,
    )
    type_metrics = {
        eval_type: attack_success_metrics(completions, ids)
        for eval_type, ids in sorted(ids_by_type.items())
    }
    category_metrics = {
        category: attack_success_metrics(completions, ids)
        for category, ids in sorted(ids_by_category.items())
    }

    summary = {
        "asr": headline_asr if headline_asr is not None else 0.0,
        "asr_ci": headline_metrics["asr_ci"],
        "n_behaviors_scored": headline_metrics["n_behaviors_scored"],
        "n_behavior_success": headline_metrics["n_behavior_success"],
        "n_behavior_failure": headline_metrics["n_behavior_failure"],
        "n_behaviors_unjudged": headline_metrics["n_behaviors_unjudged"],
        "behavior_coverage": headline_metrics["behavior_coverage"],
        "attempt_asr": headline_metrics["attempt_asr"],
        "attempt_asr_ci": headline_metrics["attempt_asr_ci"],
        "n_attempts_scored": headline_metrics["n_attempts_scored"],
        "asr_unit": headline_metrics["unit"],
        "asr_aggregation": headline_metrics["aggregation"],
        "pass_at_k": pass_curve,
        "asr_by_eval_type": {
            eval_type: values["asr"] for eval_type, values in type_metrics.items()
        },
        "n_by_eval_type": {
            eval_type: values["n_behaviors_requested"]
            for eval_type, values in type_metrics.items()
        },
        "n_scored_by_eval_type": {
            eval_type: values["n_behaviors_scored"]
            for eval_type, values in type_metrics.items()
        },
        "asr_by_category": {
            category: values["asr"] for category, values in category_metrics.items()
        },
        "macro_asr_by_category": category_groups["macro_asr"],
        "safety_worst_group_asr": category_groups["safety_worst_group_asr"],
        "safety_worst_group": category_groups["safety_worst_group"],
        "category_asr_gap": category_groups["max_min_gap"],
        "evaluation_coverage": evaluation_coverage_metrics(completions, headline_ids),
        "success_efficiency": success_efficiency_metrics(completions, headline_ids),
    }
    return completions, summary

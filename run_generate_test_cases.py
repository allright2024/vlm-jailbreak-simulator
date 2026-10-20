"""Stage 1 (unified): generate test cases for an attack over a dataset.

Mirrors generate_test_cases.py but uses the core schema/registry and writes
JSONL. Behaviors come from a registered dataset loader instead of a raw CSV, and
the output carries the behaviors used so later stages can score against the
original instruction.

Example:
    python run_generate_test_cases.py --attack FigStep --dataset SafeBench \\
        --save_dir runs/figstep_safebench --per_category 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import core.attacks  # noqa: F401  registers attacks
import core.datasets  # noqa: F401  registers datasets
from core import (
    EvalType,
    eval_type_counts,
    get_attack,
    get_dataset,
    list_attacks,
    list_datasets,
    parse_eval_type_spec,
    require_harmful_only,
    select_eval_types,
    write_jsonl,
)
from core.pipeline import build_test_cases, select_eligible
from core.config import load_attack_config
from core.data import require_data
from multimodalmodels.surrogates import get_surrogate_profile, list_surrogate_profiles


def parse_eval_types(value: str) -> tuple[EvalType, ...]:
    """Parse an explicit eval-type policy; harmful-only is the safe default."""

    try:
        return parse_eval_type_spec(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args():
    p = argparse.ArgumentParser(description="Generate jailbreak test cases (unified pipeline).")
    p.add_argument("--attack", required=True, help=f"Attack name. Available: {list_attacks()}")
    p.add_argument("--dataset", required=True, help=f"Dataset name. Available: {list_datasets()}")
    p.add_argument("--save_dir", required=True, help="Output directory for this run.")
    p.add_argument("--config", default=None, help="Attack config YAML (default: configs/attacks/<Attack>.yaml if present)")
    p.add_argument(
        "--surrogate-profile",
        choices=list_surrogate_profiles(),
        default=None,
        help=(
            "Override surrogate_backend/model/revision with a registered "
            "differentiable profile. Supported today for VisualAdv, DeltaJP, "
            "and HADES white-box."
        ),
    )
    # behavior subset selection
    p.add_argument("--limit", type=int, default=None, help="Use only the first N behaviors.")
    p.add_argument("--per_category", type=int, default=None, help="Use first N behaviors per category.")
    p.add_argument("--categories", default=None, help="Comma-separated category filter.")
    p.add_argument(
        "--eval-types",
        type=parse_eval_types,
        default=(EvalType.HARMFUL,),
        help=(
            "Comma-separated source behavior types. Default: harmful. "
            "Use 'all' only for an explicitly non-jailbreak/full-benchmark run."
        ),
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = get_dataset(args.dataset)()
    attack_cls = get_attack(args.attack)
    config = load_attack_config(attack_cls.name, args.config)
    if args.surrogate_profile:
        profile = get_surrogate_profile(args.surrogate_profile)
        if attack_cls.name not in profile.supported_attacks:
            raise ValueError(
                f"Surrogate profile {args.surrogate_profile!r} does not implement "
                f"the {attack_cls.name!r} attack contract; supported attacks: "
                f"{list(profile.supported_attacks)}"
            )
        config = {
            **config,
            "surrogate_backend": profile.backend,
            "surrogate": profile.model_id,
            "surrogate_revision": profile.revision,
        }
    # Data is distributed separately (for example via a mounted Drive). Fail
    # before constructing attacks or writing partial outputs, with every missing
    # path and the environment variable needed to resolve it.
    require_data(
        datasets=[dataset.name],
        attacks=[attack_cls.name],
        attack_configs={attack_cls.name: config},
    )
    attack = attack_cls(config)

    cats = args.categories.split(",") if args.categories else None
    # Exclude the attack's train split BEFORE applying --limit / --per_category,
    # so a small subset does not land entirely in the excluded train set (which
    # for a universal attack would yield zero eval behaviors).
    source_pool = dataset.load_subset(categories=cats)
    pool, policy_excluded = select_eval_types(source_pool, args.eval_types)
    harmful_only = set(args.eval_types) == {EvalType.HARMFUL}
    if harmful_only:
        require_harmful_only(pool)
    n_pool = len(pool)
    # train-exclusion BEFORE --limit/--per_category (they compose), so a small
    # subset is not silently emptied by a universal attack's train split.
    behaviors, excluded_behaviors = select_eligible(
        attack, pool, limit=args.limit, per_category=args.per_category
    )
    n_excluded_pre = len(excluded_behaviors)
    print(f"Attack={attack.name}  Dataset={dataset.name}  "
          f"eligible={len(behaviors)} (source-pool={len(source_pool)}, "
          f"eval-type-excluded={len(policy_excluded)}, pool={n_pool}, "
          f"train-excluded={n_excluded_pre})")

    if not behaviors:
        print(f"ERROR: no eligible behaviors for {attack.name} x {dataset.name} "
              f"after train-exclusion + subsetting (pool={n_pool}, excluded={n_excluded_pre}). "
              f"Nothing to generate; not writing outputs.")
        raise SystemExit(2)

    images_dir = os.path.join(args.save_dir, "images")
    test_cases, skipped = build_test_cases(attack, behaviors, images_dir, verbose=args.verbose)

    generated_ids = {tc.behavior_id for tc in test_cases}
    evaluated_behaviors = [b for b in behaviors if b.behavior_id in generated_ids]
    if harmful_only:
        # Fail closed immediately before persistence as a second guard against
        # future selection/refactoring mistakes.
        require_harmful_only(evaluated_behaviors)

    if not test_cases:
        print(f"ERROR: {attack.name} x {dataset.name} produced 0 test cases "
              f"(all {len(behaviors)} eligible behaviors skipped: {skipped}). "
              f"Treating as failure; not writing outputs.")
        raise SystemExit(2)

    tc_path = os.path.join(args.save_dir, "test_cases.jsonl")
    beh_path = os.path.join(args.save_dir, "behaviors.jsonl")
    write_jsonl(tc_path, test_cases)
    write_jsonl(beh_path, evaluated_behaviors)

    config_path = args.config or os.path.join(
        os.path.dirname(__file__), "configs", "attacks", f"{attack.name}.yaml"
    )
    config_sha256 = None
    if os.path.isfile(config_path):
        with open(config_path, "rb") as f:
            config_sha256 = hashlib.sha256(f.read()).hexdigest()
    all_excluded = {
        **policy_excluded,
        **excluded_behaviors,
        **{bid: "build_time" for bid in attack.excluded_behavior_ids},
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attack": attack.name,
        "dataset": dataset.name,
        **attack.provenance(),
        "config": config,
        "config_path": os.path.abspath(config_path),
        "config_sha256": config_sha256,
        "behavior_policy": "harmful_only" if harmful_only else "explicit_eval_types",
        "harmful_only": harmful_only,
        "allowed_eval_types": sorted(item.value for item in args.eval_types),
        "source_eval_type_counts": eval_type_counts(source_pool),
        "evaluated_eval_type_counts": eval_type_counts(evaluated_behaviors),
        "n_source_behaviors_after_category_filter": len(source_pool),
        "n_eval_type_excluded": len(policy_excluded),
        "n_pool_behaviors": n_pool,
        "n_train_excluded_pre_subset": n_excluded_pre,
        "n_requested_behaviors": len(behaviors),
        "n_evaluated_behaviors": len(evaluated_behaviors),
        "n_test_cases": len(test_cases),
        "training_behavior_ids": attack.training_behavior_ids(),
        "n_training_instruction_hashes": len(attack.training_instruction_hashes()),
        # train/eval exclusions with their reason, captured at the pre-subset
        # filter (plus any caught during build), so the manifest is not empty.
        "excluded_behaviors": all_excluded,
        "excluded_behavior_ids": sorted(all_excluded),
        "n_excluded_behaviors": len(all_excluded),
        "skip_counts": skipped,
    }
    manifest_path = os.path.join(args.save_dir, "run_manifest.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(test_cases)} test cases -> {tc_path}")
    print(f"Wrote {len(evaluated_behaviors)} behaviors   -> {beh_path}")
    print(f"Wrote run manifest              -> {manifest_path}")
    if skipped:
        print(f"Skipped behaviors (missing requirements): {skipped}")


if __name__ == "__main__":
    main()

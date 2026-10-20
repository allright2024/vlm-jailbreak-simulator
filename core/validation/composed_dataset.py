#!/usr/bin/env python3
"""Validate harmful-only policy, lineage, counts, and split isolation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

from core import Behavior, EvalType, TestCase, read_jsonl, require_harmful_only
from core.attack import file_sha256
from core.composition import (
    FINAL_ALGORITHM_FIDELITY,
    assign_source_split,
    composition_content_hash,
    strict_image_hashes,
    text_sha256,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate(run_dir: str) -> list[str]:
    errors: list[str] = []
    root = os.path.abspath(run_dir)
    required = {
        "manifest": os.path.join(root, "composition_manifest.json"),
        "behaviors": os.path.join(root, "behaviors.jsonl"),
        "raw_behaviors": os.path.join(root, "raw_behaviors.jsonl"),
        "final": os.path.join(root, "test_cases.jsonl"),
        "raw": os.path.join(root, "raw_test_cases.jsonl"),
    }
    for label, path in required.items():
        if not os.path.isfile(path):
            errors.append(f"missing_{label}:{path}")
    if errors:
        return errors

    manifest = _load_json(required["manifest"])
    behaviors = read_jsonl(required["behaviors"], Behavior)
    raw_behaviors = read_jsonl(required["raw_behaviors"], Behavior)
    final_cases = read_jsonl(required["final"], TestCase)
    raw_cases = read_jsonl(required["raw"], TestCase)
    try:
        require_harmful_only(behaviors)
    except ValueError as exc:
        errors.append(f"harmful_only:{exc}")
    try:
        require_harmful_only(raw_behaviors)
    except ValueError as exc:
        errors.append(f"raw_harmful_only:{exc}")
    if any(behavior.eval_type != EvalType.HARMFUL for behavior in behaviors):
        errors.append("non_harmful_behavior_present")
    behavior_by_id = {behavior.behavior_id: behavior for behavior in behaviors}
    raw_behavior_by_id = {behavior.behavior_id: behavior for behavior in raw_behaviors}
    if len(behavior_by_id) != len(behaviors):
        errors.append("duplicate_behavior_id")
    if len(raw_behavior_by_id) != len(raw_behaviors):
        errors.append("duplicate_raw_behavior_id")

    if manifest.get("algorithm_fidelity") != FINAL_ALGORITHM_FIDELITY:
        errors.append("manifest_algorithm_fidelity")
    if manifest.get("paper_reproduction") is not False:
        errors.append("manifest_overclaims_paper_reproduction")
    if manifest.get("harmful_only") is not True:
        errors.append("manifest_harmful_only_false")
    if manifest.get("selection_uses_asr") is not False:
        errors.append("selection_uses_asr_or_missing")
    if "no_asr" not in str(manifest.get("selection_policy", "")):
        errors.append("selection_policy_does_not_exclude_asr")
    if manifest.get("status") != "complete":
        errors.append(f"manifest_status:{manifest.get('status', 'missing')}")

    raw_ids = [case.test_case_id for case in raw_cases]
    final_ids = [case.test_case_id for case in final_cases]
    final_id_set = set(final_ids)
    if len(raw_ids) != len(set(raw_ids)):
        errors.append("duplicate_raw_test_case_id")
    if len(final_ids) != len(set(final_ids)):
        errors.append("duplicate_final_test_case_id")
    if not set(final_ids).issubset(raw_ids):
        errors.append("final_cases_not_subset_of_raw")

    content_hashes: set[str] = set()
    source_splits: dict[str, set[str]] = defaultdict(set)
    recipe_counts: Counter[str] = Counter()
    seed = int(manifest.get("seed", 0))
    for case in raw_cases:
        meta = case.meta
        recipe_id = str(meta.get("recipe_id", ""))
        source_id = str(meta.get("source_behavior_id", ""))
        if case.behavior_id != source_id:
            errors.append(f"{case.test_case_id}:behavior_parent_mismatch")
        if source_id not in raw_behavior_by_id:
            errors.append(f"{case.test_case_id}:missing_raw_behavior")
        if source_id not in behavior_by_id and case.test_case_id in final_id_set:
            errors.append(f"{case.test_case_id}:missing_final_behavior")
        if meta.get("algorithm_fidelity") != FINAL_ALGORITHM_FIDELITY:
            errors.append(f"{case.test_case_id}:wrong_fidelity")
        if meta.get("paper_reproduction") is not False:
            errors.append(f"{case.test_case_id}:paper_reproduction_true")
        if "no_asr" not in str(meta.get("selection_policy", "")):
            errors.append(f"{case.test_case_id}:selection_policy")
        try:
            calculated = composition_content_hash(case.text, case.images)
        except (OSError, ValueError) as exc:
            errors.append(f"{case.test_case_id}:final_artifact:{exc}")
            continue
        recorded = str(meta.get("content_hash", ""))
        if calculated != recorded:
            errors.append(f"{case.test_case_id}:content_hash_mismatch")
        if recorded in content_hashes:
            errors.append(f"{case.test_case_id}:duplicate_content_hash")
        content_hashes.add(recorded)

        split = str(meta.get("dataset_split", ""))
        source_splits[source_id].add(split)
        expected_split = assign_source_split(source_id, seed=seed)
        if split != expected_split:
            errors.append(f"{case.test_case_id}:split_hash_mismatch")

        records = meta.get("stage_records")
        arity = int(meta.get("arity", -1))
        if not isinstance(records, list) or len(records) != arity:
            errors.append(f"{case.test_case_id}:stage_count")
            continue
        previous_output = None
        for index, record in enumerate(records):
            prefix = f"{case.test_case_id}:stage{index}"
            if int(record.get("stage_index", -1)) != index:
                errors.append(f"{prefix}:index")
            if record.get("paper_reproduction") is not False:
                errors.append(f"{prefix}:paper_reproduction_true")
            if record.get("full_iterations_completed") is not True:
                errors.append(f"{prefix}:incomplete_iterations")
            if previous_output is not None and record.get("input_content_hash") != previous_output:
                errors.append(f"{prefix}:broken_continuity")
            previous_output = record.get("output_content_hash")
            for side in ("input", "output"):
                paths = record.get(f"{side}_images")
                digests = record.get(f"{side}_image_sha256s")
                if not isinstance(paths, list) or not isinstance(digests, list):
                    errors.append(f"{prefix}:{side}_image_lineage")
                    continue
                try:
                    actual = strict_image_hashes(paths)
                    if actual != digests:
                        errors.append(f"{prefix}:{side}_image_hash")
                except FileNotFoundError:
                    errors.append(f"{prefix}:{side}_image_missing")
            config_path = record.get("config_path")
            config_digest = record.get("config_sha256")
            if config_path:
                if not os.path.isfile(config_path):
                    errors.append(f"{prefix}:config_missing")
                elif file_sha256(config_path) != config_digest:
                    errors.append(f"{prefix}:config_changed")
            metadata = record.get("metadata") or {}
            candidate_paths = metadata.get("candidate_images")
            candidate_hashes = metadata.get("candidate_image_sha256s")
            if candidate_paths is not None or candidate_hashes is not None:
                if not isinstance(candidate_paths, list) or not isinstance(candidate_hashes, list):
                    errors.append(f"{prefix}:candidate_image_lineage")
                else:
                    try:
                        if strict_image_hashes(candidate_paths) != candidate_hashes:
                            errors.append(f"{prefix}:candidate_image_hash")
                    except FileNotFoundError:
                        errors.append(f"{prefix}:candidate_image_missing")
                    plan = record.get("iteration_plan") or {}
                    result = record.get("iteration_result") or {}
                    planned = plan.get("n_steps", plan.get("augmentations"))
                    completed = result.get("augmentations_completed")
                    selected_index = result.get("selected_index")
                    if planned is None or int(planned) != len(candidate_paths):
                        errors.append(f"{prefix}:candidate_plan_count")
                    if completed is None or int(completed) != len(candidate_paths):
                        errors.append(f"{prefix}:candidate_completed_count")
                    if not isinstance(selected_index, int) or not 0 <= selected_index < len(candidate_paths):
                        errors.append(f"{prefix}:candidate_selected_index")
                    elif record.get("output_images") != [candidate_paths[selected_index]]:
                        errors.append(f"{prefix}:candidate_selection_lineage")
        if records:
            first = records[0]
            source = raw_behavior_by_id.get(source_id)
            if source is not None:
                if first.get("input_text_sha256") != text_sha256(source.instruction):
                    errors.append(f"{case.test_case_id}:source_text_lineage")
                try:
                    if first.get("input_image_sha256s") != strict_image_hashes(source.images):
                        errors.append(f"{case.test_case_id}:source_image_lineage")
                except FileNotFoundError:
                    errors.append(f"{case.test_case_id}:source_image_missing")
            last = records[-1]
            if last.get("output_content_hash") != calculated:
                errors.append(f"{case.test_case_id}:final_stage_hash")
            if last.get("output_text_sha256") != text_sha256(case.text):
                errors.append(f"{case.test_case_id}:final_text_hash")
            try:
                if last.get("output_image_sha256s") != strict_image_hashes(case.images):
                    errors.append(f"{case.test_case_id}:final_image_hash")
            except FileNotFoundError:
                errors.append(f"{case.test_case_id}:final_image_missing")

    for source_id, splits in source_splits.items():
        if len(splits) != 1:
            errors.append(f"split_leakage:{source_id}:{sorted(splits)}")

    final_set = final_id_set
    for case in raw_cases:
        if case.test_case_id in final_set:
            recipe_counts[str(case.meta.get("recipe_id", ""))] += 1
            if case.meta.get("selection_state") != "selected_final":
                errors.append(f"{case.test_case_id}:final_selection_state")
        elif case.meta.get("selection_state") != "valid_reserve_not_selected":
            errors.append(f"{case.test_case_id}:reserve_selection_state")
    for recipe in manifest.get("recipes", []):
        recipe_id = str(recipe.get("recipe_id", ""))
        selected_final = int(recipe.get("selected_final", -1))
        requested_final = int(recipe.get("requested_final", -1))
        generated_raw = int(recipe.get("generated_structurally_valid", -1))
        requested_raw = int(recipe.get("requested_raw", -1))
        if recipe_counts[recipe_id] != selected_final:
            errors.append(f"recipe_count:{recipe_id}")
        if selected_final != requested_final:
            errors.append(
                f"recipe_quota:{recipe_id}:{selected_final}/{requested_final}"
            )
        if generated_raw != requested_raw:
            errors.append(
                f"recipe_raw_quota:{recipe_id}:{generated_raw}/{requested_raw}"
            )
    if len(raw_cases) != int(manifest.get("n_raw_structurally_valid", -1)):
        errors.append("manifest_raw_count")
    if len(raw_behaviors) != int(manifest.get("n_raw_unique_behaviors", -1)):
        errors.append("manifest_raw_behavior_count")
    if len(final_cases) != int(manifest.get("n_final_test_cases", -1)):
        errors.append("manifest_final_count")
    if len(content_hashes) != int(manifest.get("n_unique_content_hashes", -1)):
        errors.append("manifest_content_hash_count")
    return errors


def main() -> None:
    args = _args()
    errors = validate(args.run_dir)
    if errors:
        print(f"INVALID: {len(errors)} issue(s)", file=sys.stderr)
        for error in errors[:100]:
            print(f"  - {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} more", file=sys.stderr)
        raise SystemExit(1)
    if not args.quiet:
        print("VALID: harmful-only policy, counts, unique content, stage lineage, and splits")


if __name__ == "__main__":
    main()

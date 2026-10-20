#!/usr/bin/env python3
"""Merge complete recipe-group runs into one validated composition dataset.

The 3,000-sample production job is intentionally split across several Python
processes so model objects from earlier attack families cannot accumulate on
the GPU.  This tool performs a strict, lossless merge of those process outputs.
It rejects configuration drift, missing recipes, short raw/final quotas,
duplicate test-case/content identities, and invalid source runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from core import (
    Behavior,
    TestCase,
    eval_type_counts,
    read_jsonl,
    require_harmful_only,
    write_jsonl,
)
from core.composition import (
    FINAL_ALGORITHM_FIDELITY,
    load_composition_config,
    public_config,
    write_json,
)
from core.validation.composed_dataset import validate


OUTPUT_FILES = (
    "raw_test_cases.jsonl",
    "raw_behaviors.jsonl",
    "test_cases.jsonl",
    "behaviors.jsonl",
    "composition_qc.json",
    "generation_progress.jsonl",
    "composition_manifest.json",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="complete recipe-group run directories to merge",
    )
    return parser.parse_args()


def _read_object(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _insert_unique(
    destination: dict[str, Any], key: str, value: Any, *, label: str
) -> None:
    previous = destination.get(key)
    if previous is not None and previous.to_dict() != value.to_dict():
        raise ValueError(f"conflicting {label} for ID {key!r}")
    destination.setdefault(key, value)


def _require_unique(values: list[str], label: str) -> None:
    duplicates = [key for key, count in Counter(values).items() if count > 1]
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"duplicate {label}: {preview}")


def merge(config_path: str, output_dir: str, run_dirs: list[str]) -> str:
    config = load_composition_config(config_path)
    output = os.path.abspath(output_dir)
    sources = [os.path.abspath(path) for path in run_dirs]
    if len(sources) != len(set(sources)):
        raise ValueError("run_dirs contains duplicate paths")
    if output in sources:
        raise ValueError("output_dir must not equal a source run directory")
    existing = [name for name in OUTPUT_FILES if os.path.exists(os.path.join(output, name))]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing merged files in {output}: {existing}"
        )

    expected_order = [str(recipe["id"]) for recipe in config["recipes"]]
    expected = {str(recipe["id"]): recipe for recipe in config["recipes"]}
    recipe_manifests: dict[str, dict[str, Any]] = {}
    raw_cases: list[TestCase] = []
    final_cases: list[TestCase] = []
    raw_behaviors: dict[str, Behavior] = {}
    final_behaviors: dict[str, Behavior] = {}
    qc_records: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []

    for source in sources:
        source_errors = validate(source)
        if source_errors:
            raise ValueError(
                f"invalid source run {source}: " + "; ".join(source_errors[:10])
            )
        manifest_path = os.path.join(source, "composition_manifest.json")
        manifest = _read_object(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError(f"source run is not complete: {source}")
        if manifest.get("config_sha256") != config["_config_sha256"]:
            raise ValueError(f"composition config drift in source run: {source}")
        if int(manifest.get("seed", -1)) != int(config.get("seed", 0)):
            raise ValueError(f"selection seed drift in source run: {source}")

        source_raw = read_jsonl(os.path.join(source, "raw_test_cases.jsonl"), TestCase)
        source_final = read_jsonl(os.path.join(source, "test_cases.jsonl"), TestCase)
        source_raw_behaviors = read_jsonl(
            os.path.join(source, "raw_behaviors.jsonl"), Behavior
        )
        source_final_behaviors = read_jsonl(
            os.path.join(source, "behaviors.jsonl"), Behavior
        )
        raw_cases.extend(source_raw)
        final_cases.extend(source_final)
        for behavior in source_raw_behaviors:
            _insert_unique(
                raw_behaviors, behavior.behavior_id, behavior, label="raw behavior"
            )
        for behavior in source_final_behaviors:
            _insert_unique(
                final_behaviors, behavior.behavior_id, behavior, label="final behavior"
            )

        qc_path = os.path.join(source, "composition_qc.json")
        qc = _read_object(qc_path)
        records = qc.get("records", [])
        if not isinstance(records, list):
            raise ValueError(f"malformed QC records: {qc_path}")
        qc_records.extend(copy.deepcopy(records))

        source_recipe_ids: list[str] = []
        for recipe_manifest in manifest.get("recipes", []):
            recipe_id = str(recipe_manifest.get("recipe_id", ""))
            if recipe_id not in expected:
                raise ValueError(f"unexpected recipe {recipe_id!r} in {source}")
            if recipe_id in recipe_manifests:
                raise ValueError(f"recipe {recipe_id} appears in multiple source runs")
            recipe = expected[recipe_id]
            requested_raw = int(recipe_manifest.get("requested_raw", -1))
            requested_final = int(recipe_manifest.get("requested_final", -1))
            generated_raw = int(recipe_manifest.get("generated_structurally_valid", -1))
            selected_final = int(recipe_manifest.get("selected_final", -1))
            if requested_raw != int(recipe["raw_count"]):
                raise ValueError(
                    f"{recipe_id}: requested_raw={requested_raw}, expected={recipe['raw_count']}"
                )
            if requested_final != int(recipe["final_count"]):
                raise ValueError(
                    f"{recipe_id}: requested_final={requested_final}, "
                    f"expected={recipe['final_count']}"
                )
            if generated_raw != requested_raw:
                raise ValueError(
                    f"{recipe_id}: structurally valid raw quota {generated_raw}/{requested_raw}"
                )
            if selected_final != requested_final:
                raise ValueError(
                    f"{recipe_id}: final quota {selected_final}/{requested_final}"
                )
            if recipe_manifest.get("complete") is not True:
                raise ValueError(f"{recipe_id}: source recipe is not complete")
            recipe_manifests[recipe_id] = copy.deepcopy(recipe_manifest)
            source_recipe_ids.append(recipe_id)
        source_audit.append({
            "run_dir": source,
            "manifest": manifest_path,
            "recipe_ids": source_recipe_ids,
            "n_raw": len(source_raw),
            "n_final": len(source_final),
        })

    missing = [recipe_id for recipe_id in expected_order if recipe_id not in recipe_manifests]
    if missing:
        raise ValueError(f"missing configured recipes: {missing}")

    _require_unique([case.test_case_id for case in raw_cases], "raw test_case_id")
    _require_unique([case.test_case_id for case in final_cases], "final test_case_id")
    content_hashes = [str(case.meta.get("content_hash", "")) for case in raw_cases]
    if any(not value for value in content_hashes):
        raise ValueError("one or more raw cases have no content_hash")
    _require_unique(content_hashes, "raw content_hash")

    expected_raw_total = int(config["expected_totals"]["raw"])
    expected_final_total = int(config["expected_totals"]["final"])
    if len(raw_cases) != expected_raw_total:
        raise ValueError(f"raw total {len(raw_cases)} != configured {expected_raw_total}")
    if len(final_cases) != expected_final_total:
        raise ValueError(
            f"final total {len(final_cases)} != configured {expected_final_total}"
        )

    raw_behavior_list = list(raw_behaviors.values())
    final_behavior_list = list(final_behaviors.values())
    require_harmful_only(raw_behavior_list)
    require_harmful_only(final_behavior_list)
    os.makedirs(output, exist_ok=True)
    write_jsonl(os.path.join(output, "raw_test_cases.jsonl"), raw_cases)
    write_jsonl(os.path.join(output, "raw_behaviors.jsonl"), raw_behavior_list)
    write_jsonl(os.path.join(output, "test_cases.jsonl"), final_cases)
    write_jsonl(os.path.join(output, "behaviors.jsonl"), final_behavior_list)
    write_json(os.path.join(output, "composition_qc.json"), {
        "selection_policy": "structural_qc_then_deterministic_order_no_asr",
        "records": qc_records,
        "merged_from": source_audit,
    })
    with open(os.path.join(output, "generation_progress.jsonl"), "w", encoding="utf-8") as stream:
        for record in qc_records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    split_counts = Counter(str(case.meta.get("dataset_split", "")) for case in final_cases)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "config": public_config(config),
        "seed": int(config.get("seed", 0)),
        "algorithm_fidelity": FINAL_ALGORITHM_FIDELITY,
        "paper_reproduction": False,
        "behavior_policy": "harmful_only",
        "harmful_only": True,
        "selection_uses_asr": False,
        "selection_policy": "structural_qc_then_deterministic_order_no_asr",
        "source_group_key": "source_behavior_id",
        "split_counts": dict(sorted(split_counts.items())),
        "n_raw_structurally_valid": len(raw_cases),
        "n_raw_unique_behaviors": len(raw_behavior_list),
        "n_final_test_cases": len(final_cases),
        "n_final_unique_behaviors": len(final_behavior_list),
        "n_unique_content_hashes": len(content_hashes),
        "final_eval_type_counts": eval_type_counts(final_behavior_list),
        "raw_eval_type_counts": eval_type_counts(raw_behavior_list),
        "recipes": [recipe_manifests[recipe_id] for recipe_id in expected_order],
        "controls": copy.deepcopy(config.get("controls", {})),
        "controls_generated": False,
        "merge_sources": source_audit,
    }
    write_json(os.path.join(output, "composition_manifest.json"), manifest)

    merged_errors = validate(output)
    if merged_errors:
        raise RuntimeError(
            "merged dataset failed independent validation: "
            + "; ".join(merged_errors[:20])
        )
    return output


def main() -> None:
    args = _args()
    output = merge(args.config, args.output_dir, args.run_dirs)
    print(f"MERGED VALID: {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the configured 2/3/4-method composed jailbreak dataset.

This stage creates attack inputs only.  It never queries a victim model and
never uses ASR/judge labels to decide which candidates survive.  Run the normal
completion/evaluation stages on the resulting ``test_cases.jsonl`` afterward.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from core import (
    Behavior,
    EvalType,
    TestCase,
    eval_type_counts,
    get_dataset,
    read_jsonl,
    require_harmful_only,
    select_eval_types,
    write_jsonl,
)
from core.composition import (
    FINAL_ALGORITHM_FIDELITY,
    ComposedArtifact,
    assign_source_split,
    deterministic_stratified_select,
    execute_component,
    load_composition_config,
    public_config,
    write_json,
)
from core.composition_components import make_component
from core.validation.composed_dataset import validate

import core.datasets  # noqa: F401  (dataset registration)


DEFAULT_CONFIG = "configs/compositions/jailbreak_dataset_v3.yaml"

_METADATA_FILES = (
    "raw_test_cases.jsonl",
    "raw_behaviors.jsonl",
    "test_cases.jsonl",
    "behaviors.jsonl",
    "composition_qc.json",
    "generation_progress.jsonl",
    # The manifest is deliberately committed last: it is the transaction's
    # completion marker for readers and for the production launcher.
    "composition_manifest.json",
)
_TRANSACTION_FILE = ".composition_metadata_transaction.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--save-dir", default="runs/composed_jailbreak_v3")
    parser.add_argument(
        "--recipes",
        help="comma-separated recipe IDs (default: all configured recipes)",
    )
    parser.add_argument(
        "--per-recipe",
        type=int,
        help="pilot override for final samples per recipe (e.g. 3 -> 54 total)",
    )
    parser.add_argument("--seed", type=int, help="override global selection/split seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without loading datasets/models",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="also load datasets/configs and verify raw-parent capacity, but no models",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume a finalized incomplete bundle after strict config/lineage "
            "validation; existing valid artifacts and final selections are preserved"
        ),
    )
    return parser.parse_args()


def _selected_recipes(config: dict[str, Any], spec: str | None) -> list[dict[str, Any]]:
    recipes = list(config["recipes"])
    if not spec:
        return recipes
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    by_id = {str(recipe["id"]): recipe for recipe in recipes}
    missing = [item for item in requested if item not in by_id]
    if missing:
        raise ValueError(f"unknown recipe ID(s): {missing}; configured={sorted(by_id)}")
    return [by_id[item] for item in requested]


def _stage_key(stage_spec: dict[str, Any]) -> str:
    blob = json.dumps(stage_spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_harmful_pool(dataset_names: Iterable[str]) -> tuple[list[Behavior], dict[str, Any]]:
    source: list[Behavior] = []
    dataset_counts: dict[str, int] = {}
    excluded_counts: Counter[str] = Counter()
    for name in dataset_names:
        loaded = get_dataset(str(name))().load()
        dataset_counts[str(name)] = len(loaded)
        selected, excluded = select_eval_types(loaded, (EvalType.HARMFUL,))
        source.extend(selected)
        excluded_counts.update(excluded.values())
    unique: dict[str, Behavior] = {}
    for behavior in source:
        previous = unique.get(behavior.behavior_id)
        if previous is not None and previous.to_dict() != behavior.to_dict():
            raise ValueError(
                f"conflicting source behavior ID across datasets: {behavior.behavior_id}"
            )
        unique.setdefault(behavior.behavior_id, behavior)
    pool = list(unique.values())
    require_harmful_only(pool)
    return pool, {
        "source_dataset_counts": dataset_counts,
        "source_eval_type_counts": eval_type_counts(source),
        "policy_excluded_counts": dict(sorted(excluded_counts.items())),
        "harmful_unique_count": len(pool),
    }


def _eligible_for_all(components, pool: list[Behavior]):
    eligible: list[Behavior] = []
    exclusions: dict[str, list[str]] = {}
    for behavior in pool:
        reasons = []
        for component in components:
            reason = component.evaluation_exclusion_reason(behavior)
            if reason:
                reasons.append(f"{component.component_name}:{reason}")
        if reasons:
            exclusions[behavior.behavior_id] = reasons
        else:
            eligible.append(behavior)
    return eligible, exclusions


def _structural_errors(
    artifact: ComposedArtifact, behavior: Behavior, *, arity: int
) -> list[str]:
    errors: list[str] = []
    if behavior.eval_type != EvalType.HARMFUL:
        errors.append("non_harmful_parent")
    if artifact.source_behavior_id != behavior.behavior_id:
        errors.append("missing_or_wrong_parent")
    if not artifact.text.strip():
        errors.append("empty_text")
    if not artifact.images:
        errors.append("missing_final_image")
    if len(artifact.stage_records) != arity:
        errors.append("incomplete_stage_count")
    if any(not record.full_iterations_completed for record in artifact.stage_records):
        errors.append("incomplete_iterations")
    for previous, current in zip(artifact.stage_records, artifact.stage_records[1:]):
        if previous.output_content_hash != current.input_content_hash:
            errors.append("broken_stage_continuity")
            break
    if artifact.stage_records:
        try:
            if artifact.stage_records[-1].output_content_hash != artifact.content_hash:
                errors.append("final_hash_mismatch")
        except (OSError, ValueError):
            errors.append("missing_or_unreadable_image")
    return errors


def _case_from_artifact(
    artifact: ComposedArtifact,
    behavior: Behavior,
    recipe: dict[str, Any],
    *,
    rank: int,
    seed: int,
) -> TestCase:
    recipe_id = str(recipe["id"])
    split = assign_source_split(behavior.behavior_id, seed=seed)
    content_hash = artifact.content_hash
    return TestCase(
        test_case_id=f"Composed:{recipe_id}:{behavior.behavior_id}:0",
        behavior_id=behavior.behavior_id,
        attack=f"COMPOSED-{recipe_id}",
        text=artifact.text,
        images=list(artifact.images),
        attempt=0,
        meta={
            "composition_schema_version": 1,
            "recipe_id": recipe_id,
            "recipe_name": recipe["name"],
            "arity": int(recipe["arity"]),
            "source_behavior_id": behavior.behavior_id,
            "source_dataset": behavior.dataset,
            "dataset_split": split,
            "content_hash": content_hash,
            "candidate_rank": rank,
            "selection_state": "raw_structurally_valid",
            "selection_policy": "structural_qc_then_deterministic_order_no_asr",
            "algorithm_fidelity": FINAL_ALGORITHM_FIDELITY,
            "paper_reproduction": False,
            "stage_records": [record.to_dict() for record in artifact.stage_records],
        },
    )


def _append_progress(path: str, record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json_object(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _candidate_seed(recipe_id: str, seed: int) -> int:
    return seed + int.from_bytes(recipe_id.encode(), "little")


def _deterministic_candidate_order(
    eligible: list[Behavior], recipe_id: str, seed: int
) -> list[Behavior]:
    """Return the full stable candidate stream used for QC/backfill.

    ``deterministic_stratified_select`` is prefix-stable, so the first
    ``raw_count`` entries are byte-for-byte the same parents selected by the
    original generator.  Structural failures are backfilled only from later
    ranks and never change an already selected final prefix.
    """

    return deterministic_stratified_select(
        eligible, len(eligible), seed=_candidate_seed(recipe_id, seed)
    )


def _failure_key(record: dict[str, Any]) -> str | None:
    if record.get("status") != "rejected_structural_qc":
        return None
    reasons = record.get("reasons")
    if not isinstance(reasons, list) or not reasons or not all(
        isinstance(reason, str) and reason for reason in reasons
    ):
        raise ValueError(f"malformed rejected QC record: {record!r}")
    return ",".join(reasons)


def _resume_validation_error_allowed(error: str) -> bool:
    return error == "manifest_status:incomplete" or error.startswith(
        ("recipe_raw_quota:", "recipe_quota:")
    )


def _fsync_file(path: str) -> None:
    with open(path, "rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: str, value: dict[str, Any]) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    temporary = os.path.join(
        parent, f".{os.path.basename(path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finish_pending_metadata_commit(save_dir: str) -> bool:
    """Roll forward a fully staged metadata transaction after interruption."""

    marker_path = os.path.join(save_dir, _TRANSACTION_FILE)
    if not os.path.isfile(marker_path):
        return False
    marker = _read_json_object(marker_path)
    staging_name = marker.get("staging_dir")
    files = marker.get("files")
    if (
        not isinstance(staging_name, str)
        or os.path.basename(staging_name) != staging_name
        or not staging_name.startswith(".composition_metadata_staging_")
        or files != list(_METADATA_FILES)
    ):
        raise RuntimeError(f"unsafe or malformed metadata transaction: {marker_path}")
    staging_dir = os.path.join(save_dir, staging_name)
    for filename in _METADATA_FILES:
        staged = os.path.join(staging_dir, filename)
        destination = os.path.join(save_dir, filename)
        if os.path.isfile(staged):
            os.replace(staged, destination)
            _fsync_directory(save_dir)
        elif not os.path.isfile(destination):
            raise RuntimeError(
                f"metadata transaction lost both staged and destination {filename}"
            )
    os.unlink(marker_path)
    if os.path.isdir(staging_dir):
        os.rmdir(staging_dir)
    _fsync_directory(save_dir)
    print(f"Recovered pending metadata commit in {save_dir}")
    return True


def _backup_metadata(save_dir: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup_dir = os.path.join(save_dir, ".resume_backups", timestamp)
    os.makedirs(backup_dir, exist_ok=False)
    for filename in _METADATA_FILES:
        source = os.path.join(save_dir, filename)
        if os.path.isfile(source):
            destination = os.path.join(backup_dir, filename)
            shutil.copy2(source, destination)
            _fsync_file(destination)
    _fsync_directory(backup_dir)
    _fsync_directory(os.path.dirname(backup_dir))
    return backup_dir


def _commit_metadata(
    save_dir: str,
    *,
    all_raw_cases: list[TestCase],
    raw_behaviors: list[Behavior],
    all_final_cases: list[TestCase],
    final_behaviors: list[Behavior],
    qc_records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """Stage all metadata, then atomically replace files with manifest last.

    The transaction marker is installed only after every staged file is fully
    written and fsynced.  A later ``--resume`` can safely roll a partially
    committed transaction forward without re-running an attack component.
    """

    staging_name = f".composition_metadata_staging_{os.getpid()}_{uuid.uuid4().hex}"
    staging_dir = os.path.join(save_dir, staging_name)
    os.makedirs(staging_dir, exist_ok=False)
    write_jsonl(os.path.join(staging_dir, "raw_test_cases.jsonl"), all_raw_cases)
    write_jsonl(os.path.join(staging_dir, "raw_behaviors.jsonl"), raw_behaviors)
    write_jsonl(os.path.join(staging_dir, "test_cases.jsonl"), all_final_cases)
    write_jsonl(os.path.join(staging_dir, "behaviors.jsonl"), final_behaviors)
    write_json(
        os.path.join(staging_dir, "composition_qc.json"),
        {
            "selection_policy": "structural_qc_then_deterministic_order_no_asr",
            "records": qc_records,
        },
    )
    with open(
        os.path.join(staging_dir, "generation_progress.jsonl"),
        "w",
        encoding="utf-8",
    ) as stream:
        for record in qc_records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json(os.path.join(staging_dir, "composition_manifest.json"), manifest)
    for filename in _METADATA_FILES:
        _fsync_file(os.path.join(staging_dir, filename))
    _fsync_directory(staging_dir)

    marker_path = os.path.join(save_dir, _TRANSACTION_FILE)
    _atomic_write_json(
        marker_path,
        {"schema_version": 1, "staging_dir": staging_name, "files": list(_METADATA_FILES)},
    )
    _finish_pending_metadata_commit(save_dir)


def _load_resume_bundle(save_dir: str) -> dict[str, Any]:
    required = set(_METADATA_FILES)
    missing = sorted(
        filename for filename in required if not os.path.isfile(os.path.join(save_dir, filename))
    )
    if missing:
        raise RuntimeError(
            "--resume requires a finalized incomplete bundle; missing: "
            + ", ".join(missing)
        )
    manifest = _read_json_object(os.path.join(save_dir, "composition_manifest.json"))
    errors = validate(save_dir)
    if manifest.get("status") == "complete":
        if errors:
            raise RuntimeError(
                "refusing to resume a corrupt complete bundle: " + "; ".join(errors[:20])
            )
        return {"already_complete": True, "manifest": manifest}
    if manifest.get("status") != "incomplete":
        raise RuntimeError(
            f"--resume requires manifest status incomplete, got {manifest.get('status')!r}"
        )
    unexpected = [error for error in errors if not _resume_validation_error_allowed(error)]
    if unexpected:
        raise RuntimeError(
            "refusing to resume bundle with integrity errors: "
            + "; ".join(unexpected[:20])
        )
    qc = _read_json_object(os.path.join(save_dir, "composition_qc.json"))
    records = qc.get("records")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise RuntimeError("composition_qc.json records must be a list of objects")
    return {
        "already_complete": False,
        "manifest": manifest,
        "raw_cases": read_jsonl(os.path.join(save_dir, "raw_test_cases.jsonl"), TestCase),
        "final_cases": read_jsonl(os.path.join(save_dir, "test_cases.jsonl"), TestCase),
        "raw_behaviors": read_jsonl(
            os.path.join(save_dir, "raw_behaviors.jsonl"), Behavior
        ),
        "final_behaviors": read_jsonl(os.path.join(save_dir, "behaviors.jsonl"), Behavior),
        "qc_records": list(records),
        "validation_errors": errors,
    }


def _print_plan(recipes: list[dict[str, Any]], per_recipe: int | None, multiplier: float) -> None:
    final_total = raw_total = 0
    for recipe in recipes:
        final_count = int(per_recipe if per_recipe is not None else recipe["final_count"])
        raw_count = (
            math.ceil(final_count * multiplier)
            if per_recipe is not None
            else int(recipe["raw_count"])
        )
        final_total += final_count
        raw_total += raw_count
        stages = " -> ".join(stage["component"] for stage in recipe["stages"])
        print(
            f"{recipe['id']}: final={final_count} raw={raw_count} "
            f"datasets={','.join(recipe['datasets'])} stages={stages}"
        )
    print(f"TOTAL: final={final_total} raw={raw_total}")


def _legacy_main() -> None:
    args = _parse_args()
    if args.per_recipe is not None and args.per_recipe <= 0:
        raise SystemExit("--per-recipe must be positive")
    config = load_composition_config(args.config)
    recipes = _selected_recipes(config, args.recipes)
    multiplier = float(config.get("candidate_multiplier", 1.2))
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    _print_plan(recipes, args.per_recipe, multiplier)
    if args.dry_run:
        return
    if args.preflight:
        pool_components: dict[str, Any] = {}
        failures = []
        for recipe in recipes:
            final_count = int(
                args.per_recipe if args.per_recipe is not None else recipe["final_count"]
            )
            raw_count = (
                math.ceil(final_count * multiplier)
                if args.per_recipe is not None
                else int(recipe["raw_count"])
            )
            try:
                components = []
                for stage_spec in recipe["stages"]:
                    key = _stage_key(stage_spec)
                    if key not in pool_components:
                        component = make_component(stage_spec)
                        component.preflight()
                        pool_components[key] = component
                    components.append(pool_components[key])
                pool, _ = _load_harmful_pool(recipe["datasets"])
                eligible, excluded = _eligible_for_all(components, pool)
            except Exception as exc:
                failures.append(str(recipe["id"]))
                print(
                    f"PREFLIGHT {recipe['id']}: ERROR {type(exc).__name__}: {exc}"
                )
                continue
            ok = len(eligible) >= raw_count
            if not ok:
                failures.append(str(recipe["id"]))
            print(
                f"PREFLIGHT {recipe['id']}: harmful={len(pool)} "
                f"train-excluded={len(excluded)} eligible={len(eligible)} "
                f"required-raw={raw_count} {'OK' if ok else 'INSUFFICIENT'}"
            )
        if failures:
            raise SystemExit(f"preflight failed for recipes: {','.join(failures)}")
        print("PREFLIGHT OK: all recipes have enough harmful-only eligible parents")
        return

    save_dir = os.path.abspath(args.save_dir)
    images_root = os.path.join(save_dir, "images")
    progress_path = os.path.join(save_dir, "generation_progress.jsonl")
    os.makedirs(images_root, exist_ok=True)
    os.environ.setdefault("VLM_COMPOSITION_CACHE_ROOT", os.path.join(save_dir, "_universal_cache"))
    # A fresh invocation gets an unambiguous progress journal.  Generated image
    # artifacts are never deleted or overwritten outside this explicit run dir.
    with open(progress_path, "w", encoding="utf-8"):
        pass

    component_pool: dict[str, Any] = {}
    prepared: set[str] = set()
    all_raw_cases: list[TestCase] = []
    all_final_cases: list[TestCase] = []
    raw_behaviors: dict[str, Behavior] = {}
    final_behaviors: dict[str, Behavior] = {}
    seen_content_hashes: set[str] = set()
    qc_records: list[dict[str, Any]] = []
    recipe_manifests: list[dict[str, Any]] = []
    incomplete = False

    for recipe in recipes:
        recipe_id = str(recipe["id"])
        final_count = int(args.per_recipe if args.per_recipe is not None else recipe["final_count"])
        raw_count = (
            math.ceil(final_count * multiplier)
            if args.per_recipe is not None
            else int(recipe["raw_count"])
        )
        components = []
        component_keys = []
        for stage_spec in recipe["stages"]:
            key = _stage_key(stage_spec)
            component = component_pool.get(key)
            if component is None:
                component = make_component(stage_spec)
                component_pool[key] = component
            components.append(component)
            component_keys.append(key)

        pool, pool_audit = _load_harmful_pool(recipe["datasets"])
        eligible, train_exclusions = _eligible_for_all(components, pool)
        selected = deterministic_stratified_select(
            eligible, raw_count, seed=seed + int.from_bytes(recipe_id.encode(), "little")
        )
        if len(selected) < raw_count:
            raise RuntimeError(
                f"{recipe_id}: requested {raw_count} raw harmful parents but only "
                f"{len(eligible)} are eligible after train exclusion"
            )

        for component, key in zip(components, component_keys):
            if key not in prepared:
                shared_dir = os.path.join(images_root, "_shared", key)
                if args.verbose:
                    print(f"[{recipe_id}] prepare {component.component_name}")
                prepare_started = time.perf_counter()
                component.prepare(shared_dir)
                component.prepare_seconds = time.perf_counter() - prepare_started
                prepared.add(key)

        recipe_valid: list[TestCase] = []
        recipe_failures: Counter[str] = Counter()
        for rank, behavior in enumerate(selected):
            artifact = ComposedArtifact(
                source_behavior_id=behavior.behavior_id,
                text=behavior.instruction,
                images=list(behavior.images),
                metadata={"source_dataset": behavior.dataset},
            )
            candidate_root = os.path.join(
                images_root, recipe_id, f"{rank:04d}_{hashlib.sha256(behavior.behavior_id.encode()).hexdigest()[:10]}"
            )
            try:
                for stage_index, component in enumerate(components):
                    artifact = execute_component(
                        component,
                        artifact,
                        behavior,
                        os.path.join(candidate_root, f"stage_{stage_index + 1:02d}_{component.component_name}"),
                        stage_index=stage_index,
                    )
                errors = _structural_errors(
                    artifact, behavior, arity=int(recipe["arity"])
                )
                if errors:
                    reason = ",".join(errors)
                    recipe_failures[reason] += 1
                    qc_records.append({
                        "recipe_id": recipe_id,
                        "source_behavior_id": behavior.behavior_id,
                        "candidate_rank": rank,
                        "status": "rejected_structural_qc",
                        "reasons": errors,
                    })
                    _append_progress(progress_path, qc_records[-1])
                    continue
                content_hash = artifact.content_hash
                if content_hash in seen_content_hashes:
                    recipe_failures["duplicate_content_hash"] += 1
                    record = {
                        "recipe_id": recipe_id,
                        "source_behavior_id": behavior.behavior_id,
                        "candidate_rank": rank,
                        "status": "rejected_structural_qc",
                        "reasons": ["duplicate_content_hash"],
                        "content_hash": content_hash,
                    }
                    qc_records.append(record)
                    _append_progress(progress_path, record)
                    continue
                seen_content_hashes.add(content_hash)
                case = _case_from_artifact(
                    artifact, behavior, recipe, rank=rank, seed=seed
                )
                recipe_valid.append(case)
                all_raw_cases.append(case)
                raw_behaviors.setdefault(behavior.behavior_id, behavior)
                record = {
                    "recipe_id": recipe_id,
                    "source_behavior_id": behavior.behavior_id,
                    "candidate_rank": rank,
                    "status": "raw_structurally_valid",
                    "content_hash": content_hash,
                }
                qc_records.append(record)
                _append_progress(progress_path, record)
                if args.verbose:
                    print(f"[{recipe_id}] {rank + 1}/{raw_count} valid {behavior.behavior_id}")
            except Exception as exc:  # keep other raw candidates; audit exact failure
                reason = f"stage_exception:{type(exc).__name__}"
                recipe_failures[reason] += 1
                record = {
                    "recipe_id": recipe_id,
                    "source_behavior_id": behavior.behavior_id,
                    "candidate_rank": rank,
                    "status": "rejected_structural_qc",
                    "reasons": [reason],
                    "error": str(exc)[:1000],
                }
                qc_records.append(record)
                _append_progress(progress_path, record)
                if args.verbose:
                    print(f"[{recipe_id}] reject {behavior.behavior_id}: {reason}")

        selected_final = recipe_valid[:final_count]
        for case in selected_final:
            case.meta["selection_state"] = "selected_final"
            all_final_cases.append(case)
            final_behaviors.setdefault(case.behavior_id, next(
                behavior for behavior in selected if behavior.behavior_id == case.behavior_id
            ))
        for case in recipe_valid[final_count:]:
            case.meta["selection_state"] = "valid_reserve_not_selected"

        # The configured reserve is part of the dataset contract: a run is
        # complete only when both the raw structurally-valid quota and the
        # selected-final quota are exact. This prevents a nominally complete
        # 3,000-final run from silently containing fewer than 3,600 raw rows.
        enough = len(recipe_valid) == raw_count and len(selected_final) == final_count
        incomplete = incomplete or not enough
        recipe_manifests.append({
            "recipe_id": recipe_id,
            "name": recipe["name"],
            "arity": recipe["arity"],
            "datasets": list(recipe["datasets"]),
            "requested_raw": raw_count,
            "requested_final": final_count,
            "generated_structurally_valid": len(recipe_valid),
            "selected_final": len(selected_final),
            "valid_reserve": max(0, len(recipe_valid) - final_count),
            "complete": enough,
            "failure_counts": dict(sorted(recipe_failures.items())),
            "n_train_excluded": len(train_exclusions),
            "train_exclusions": train_exclusions,
            "pool_audit": pool_audit,
            "components": [component.provenance() for component in components],
        })
        print(
            f"[{recipe_id}] structurally-valid={len(recipe_valid)}/{raw_count}, "
            f"final={len(selected_final)}/{final_count}"
        )

    final_behavior_list = list(final_behaviors.values())
    raw_behavior_list = list(raw_behaviors.values())
    require_harmful_only(raw_behavior_list)
    require_harmful_only(final_behavior_list)
    write_jsonl(os.path.join(save_dir, "raw_test_cases.jsonl"), all_raw_cases)
    write_jsonl(os.path.join(save_dir, "raw_behaviors.jsonl"), raw_behavior_list)
    write_jsonl(os.path.join(save_dir, "test_cases.jsonl"), all_final_cases)
    write_jsonl(os.path.join(save_dir, "behaviors.jsonl"), final_behavior_list)
    write_json(os.path.join(save_dir, "composition_qc.json"), {
        "selection_policy": "structural_qc_then_deterministic_order_no_asr",
        "records": qc_records,
    })

    split_counts = Counter(case.meta["dataset_split"] for case in all_final_cases)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "incomplete" if incomplete else "complete",
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "config": public_config(config),
        "seed": seed,
        "algorithm_fidelity": FINAL_ALGORITHM_FIDELITY,
        "paper_reproduction": False,
        "behavior_policy": "harmful_only",
        "harmful_only": True,
        "selection_uses_asr": False,
        "selection_policy": "structural_qc_then_deterministic_order_no_asr",
        "source_group_key": "source_behavior_id",
        "split_counts": dict(sorted(split_counts.items())),
        "n_raw_structurally_valid": len(all_raw_cases),
        "n_raw_unique_behaviors": len(raw_behavior_list),
        "n_final_test_cases": len(all_final_cases),
        "n_final_unique_behaviors": len(final_behavior_list),
        "n_unique_content_hashes": len({case.meta["content_hash"] for case in all_raw_cases}),
        "final_eval_type_counts": eval_type_counts(final_behavior_list),
        "raw_eval_type_counts": eval_type_counts(raw_behavior_list),
        "recipes": recipe_manifests,
        "controls": copy.deepcopy(config.get("controls", {})),
        "controls_generated": False,
    }
    write_json(os.path.join(save_dir, "composition_manifest.json"), manifest)
    print(f"Wrote {len(all_final_cases)} final cases -> {save_dir}/test_cases.jsonl")
    print(f"Wrote {len(all_raw_cases)} raw valid cases -> {save_dir}/raw_test_cases.jsonl")
    if incomplete:
        print(
            "ERROR: at least one recipe did not reach its configured raw/final quota",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _checkpoint_contexts(
    config: dict[str, Any], contexts: list[dict[str, Any]], save_dir: str,
    *, seed: int, existing_manifest: dict[str, Any], backup_dir: str | None,
) -> None:
    """Persist a resumable snapshot, including unfinished and future recipes.

    Store actual TestCases and their stage records, not just image counts.
    The same transactional commit and strict resume validator as final output
    protect periodic snapshots from a partial multi-file write.
    """

    raw_cases: list[TestCase] = []
    final_cases: list[TestCase] = []
    raw_parents: dict[str, Behavior] = {}
    final_parents: dict[str, Behavior] = {}
    qc: list[dict[str, Any]] = []
    recipe_manifests = []
    old_recipes = {r["recipe_id"]: r for r in existing_manifest.get("recipes", [])}
    complete = True
    for context in contexts:
        recipe = context["recipe"]
        cases = sorted(context["recipe_valid"], key=lambda case: int(case.meta["candidate_rank"]))
        parents = {b.behavior_id: b for b in context["pool"]}
        final_count, raw_count = context["final_count"], context["raw_count"]
        for rank, original in enumerate(cases):
            case = copy.deepcopy(original)
            selected = rank < final_count
            case.meta["selection_state"] = "selected_final" if selected else "valid_reserve_not_selected"
            raw_cases.append(case)
            raw_parents.setdefault(case.behavior_id, parents[case.behavior_id])
            if selected:
                final_cases.append(case)
                final_parents.setdefault(case.behavior_id, parents[case.behavior_id])
        records = context["recipe_qc"]
        qc.extend(copy.deepcopy(records))
        enough = len(cases) == raw_count
        complete = complete and enough
        failures = Counter(key for key in (_failure_key(r) for r in records) if key)
        if recipe["id"] in old_recipes and not any(c._prepared for c in context["components"]):
            provenance = copy.deepcopy(old_recipes[recipe["id"]].get("components", []))
        else:
            provenance = [c.provenance() for c in context["components"]]
        recipe_manifests.append({
            "recipe_id": recipe["id"], "name": recipe["name"], "arity": recipe["arity"],
            "datasets": list(recipe["datasets"]), "requested_raw": raw_count,
            "requested_final": final_count, "generated_structurally_valid": len(cases),
            "selected_final": min(len(cases), final_count),
            "valid_reserve": max(0, len(cases) - final_count), "complete": enough,
            "attempted_candidates": len(records),
            "backfilled_candidates": max(0, len(records) - raw_count),
            "candidate_pool_size": len(context["candidate_order"]),
            "failure_counts": dict(sorted(failures.items())),
            "n_train_excluded": len(context["train_exclusions"]),
            "train_exclusions": context["train_exclusions"],
            "pool_audit": context["pool_audit"], "components": provenance,
        })
    raw_behaviors, final_behaviors = list(raw_parents.values()), list(final_parents.values())
    require_harmful_only(raw_behaviors)
    require_harmful_only(final_behaviors)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1, "created_at": existing_manifest.get("created_at", now),
        "updated_at": now, "status": "complete" if complete else "incomplete",
        "config_path": config["_config_path"], "config_sha256": config["_config_sha256"],
        "config": public_config(config), "seed": seed,
        "algorithm_fidelity": FINAL_ALGORITHM_FIDELITY, "paper_reproduction": False,
        "behavior_policy": "harmful_only", "harmful_only": True,
        "selection_uses_asr": False,
        "selection_policy": "structural_qc_then_deterministic_order_no_asr",
        "source_group_key": "source_behavior_id",
        "split_counts": dict(sorted(Counter(c.meta["dataset_split"] for c in final_cases).items())),
        "n_raw_structurally_valid": len(raw_cases), "n_raw_unique_behaviors": len(raw_behaviors),
        "n_final_test_cases": len(final_cases), "n_final_unique_behaviors": len(final_behaviors),
        "n_unique_content_hashes": len({c.meta["content_hash"] for c in raw_cases}),
        "final_eval_type_counts": eval_type_counts(final_behaviors),
        "raw_eval_type_counts": eval_type_counts(raw_behaviors), "recipes": recipe_manifests,
        "controls": copy.deepcopy(config.get("controls", {})), "controls_generated": False,
        "checkpoint": {"periodic": True, "every_candidates": config.get("checkpoint_every", 10)},
    }
    if backup_dir is not None:
        manifest["resume"] = {
            "resumed": True, "metadata_backup": backup_dir,
            "resume_count": int((existing_manifest.get("resume") or {}).get("resume_count", 0)) + 1,
            "prior_status": existing_manifest.get("status"),
        }
    _commit_metadata(
        save_dir, all_raw_cases=raw_cases, raw_behaviors=raw_behaviors,
        all_final_cases=final_cases, final_behaviors=final_behaviors,
        qc_records=qc, manifest=manifest,
    )


def _main_resumable() -> None:
    """Generate a fresh bundle or safely backfill a finalized incomplete one."""

    args = _parse_args()
    if args.per_recipe is not None and args.per_recipe <= 0:
        raise SystemExit("--per-recipe must be positive")
    if args.resume and (args.dry_run or args.preflight):
        raise SystemExit("--resume cannot be combined with --dry-run or --preflight")
    config = load_composition_config(args.config)
    recipes = _selected_recipes(config, args.recipes)
    multiplier = float(config.get("candidate_multiplier", 1.2))
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    _print_plan(recipes, args.per_recipe, multiplier)
    if args.dry_run:
        return
    if args.preflight:
        pool_components: dict[str, Any] = {}
        failures: list[str] = []
        for recipe in recipes:
            final_count = int(
                args.per_recipe if args.per_recipe is not None else recipe["final_count"]
            )
            raw_count = (
                math.ceil(final_count * multiplier)
                if args.per_recipe is not None
                else int(recipe["raw_count"])
            )
            try:
                components = []
                for stage_spec in recipe["stages"]:
                    key = _stage_key(stage_spec)
                    if key not in pool_components:
                        component = make_component(stage_spec)
                        component.preflight()
                        pool_components[key] = component
                    components.append(pool_components[key])
                pool, _ = _load_harmful_pool(recipe["datasets"])
                eligible, excluded = _eligible_for_all(components, pool)
            except Exception as exc:
                failures.append(str(recipe["id"]))
                print(f"PREFLIGHT {recipe['id']}: ERROR {type(exc).__name__}: {exc}")
                continue
            ok = len(eligible) >= raw_count
            if not ok:
                failures.append(str(recipe["id"]))
            print(
                f"PREFLIGHT {recipe['id']}: harmful={len(pool)} "
                f"train-excluded={len(excluded)} eligible={len(eligible)} "
                f"required-raw={raw_count} {'OK' if ok else 'INSUFFICIENT'}"
            )
        if failures:
            raise SystemExit(f"preflight failed for recipes: {','.join(failures)}")
        print("PREFLIGHT OK: all recipes have enough harmful-only eligible parents")
        return

    save_dir = os.path.abspath(args.save_dir)
    images_root = os.path.join(save_dir, "images")
    progress_path = os.path.join(save_dir, "generation_progress.jsonl")
    resume_state: dict[str, Any] | None = None
    if args.resume:
        if not os.path.isdir(save_dir):
            raise SystemExit(f"--resume directory does not exist: {save_dir}")
        _finish_pending_metadata_commit(save_dir)
        resume_state = _load_resume_bundle(save_dir)
        if resume_state["already_complete"]:
            print(f"Resume no-op: bundle is already complete and valid: {save_dir}")
            return
    elif os.path.isdir(save_dir) and os.listdir(save_dir):
        raise SystemExit(
            f"refusing to overwrite non-empty save directory: {save_dir}; "
            "use --resume only for a finalized incomplete bundle"
        )

    os.makedirs(images_root, exist_ok=True)
    os.environ.setdefault("VLM_COMPOSITION_CACHE_ROOT", os.path.join(save_dir, "_universal_cache"))
    if not args.resume:
        with open(progress_path, "w", encoding="utf-8") as stream:
            stream.flush()
            os.fsync(stream.fileno())

    def recipe_counts(recipe: dict[str, Any]) -> tuple[int, int]:
        final_count = int(
            args.per_recipe if args.per_recipe is not None else recipe["final_count"]
        )
        raw_count = (
            math.ceil(final_count * multiplier)
            if args.per_recipe is not None
            else int(recipe["raw_count"])
        )
        return final_count, raw_count

    existing_manifest: dict[str, Any] = {}
    existing_manifest_by_recipe: dict[str, dict[str, Any]] = {}
    existing_cases_by_recipe: dict[str, list[TestCase]] = defaultdict(list)
    existing_final_by_recipe: dict[str, list[TestCase]] = defaultdict(list)
    existing_qc_by_recipe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    existing_raw_behavior_by_id: dict[str, Behavior] = {}
    qc_records: list[dict[str, Any]] = []
    seen_content_hashes: set[str] = set()
    expected_recipe_ids = [str(recipe["id"]) for recipe in recipes]

    if resume_state is not None:
        existing_manifest = resume_state["manifest"]
        if existing_manifest.get("config_sha256") != config["_config_sha256"]:
            raise RuntimeError("refusing resume: composition config SHA-256 changed")
        if int(existing_manifest.get("seed", -1)) != seed:
            raise RuntimeError("refusing resume: selection seed changed")
        old_recipe_manifests = existing_manifest.get("recipes")
        if not isinstance(old_recipe_manifests, list):
            raise RuntimeError("refusing resume: manifest recipes is malformed")
        old_recipe_ids = [str(item.get("recipe_id", "")) for item in old_recipe_manifests]
        if old_recipe_ids != expected_recipe_ids:
            raise RuntimeError(
                f"refusing resume: recipe order changed {old_recipe_ids} != {expected_recipe_ids}"
            )
        existing_manifest_by_recipe = {
            str(item["recipe_id"]): item for item in old_recipe_manifests
        }
        for recipe in recipes:
            recipe_id = str(recipe["id"])
            final_count, raw_count = recipe_counts(recipe)
            old = existing_manifest_by_recipe[recipe_id]
            if int(old.get("requested_raw", -1)) != raw_count or int(
                old.get("requested_final", -1)
            ) != final_count:
                raise RuntimeError(f"refusing resume: quota changed for {recipe_id}")
        for case in resume_state["raw_cases"]:
            recipe_id = str(case.meta.get("recipe_id", ""))
            if recipe_id not in expected_recipe_ids:
                raise RuntimeError(f"unexpected raw case recipe during resume: {recipe_id}")
            existing_cases_by_recipe[recipe_id].append(case)
            content_hash = str(case.meta.get("content_hash", ""))
            if not content_hash or content_hash in seen_content_hashes:
                raise RuntimeError("missing or duplicate existing content hash")
            seen_content_hashes.add(content_hash)
        for case in resume_state["final_cases"]:
            recipe_id = str(case.meta.get("recipe_id", ""))
            if recipe_id not in expected_recipe_ids:
                raise RuntimeError(f"unexpected final case recipe during resume: {recipe_id}")
            existing_final_by_recipe[recipe_id].append(case)
        for record in resume_state["qc_records"]:
            recipe_id = str(record.get("recipe_id", ""))
            if recipe_id not in expected_recipe_ids:
                raise RuntimeError(f"unexpected QC recipe during resume: {recipe_id}")
            existing_qc_by_recipe[recipe_id].append(record)
        qc_records = list(resume_state["qc_records"])
        for behavior in resume_state["raw_behaviors"]:
            if behavior.behavior_id in existing_raw_behavior_by_id:
                raise RuntimeError(f"duplicate raw behavior ID: {behavior.behavior_id}")
            existing_raw_behavior_by_id[behavior.behavior_id] = behavior
        raw_case_behavior_ids = {
            case.behavior_id for cases in existing_cases_by_recipe.values() for case in cases
        }
        if set(existing_raw_behavior_by_id) != raw_case_behavior_ids:
            raise RuntimeError("existing raw behavior rows do not match raw test cases")
        final_behavior_ids = {behavior.behavior_id for behavior in resume_state["final_behaviors"]}
        final_case_behavior_ids = {
            case.behavior_id for cases in existing_final_by_recipe.values() for case in cases
        }
        if final_behavior_ids != final_case_behavior_ids:
            raise RuntimeError("existing final behavior rows do not match final test cases")

    component_pool: dict[str, Any] = {}
    contexts: list[dict[str, Any]] = []
    # Validate every recipe and deterministic rank before any model is loaded.
    for recipe in recipes:
        recipe_id = str(recipe["id"])
        final_count, raw_count = recipe_counts(recipe)
        components = []
        component_keys = []
        for stage_spec in recipe["stages"]:
            key = _stage_key(stage_spec)
            component = component_pool.get(key)
            if component is None:
                component = make_component(stage_spec)
                component_pool[key] = component
            components.append(component)
            component_keys.append(key)
        pool, pool_audit = _load_harmful_pool(recipe["datasets"])
        eligible, train_exclusions = _eligible_for_all(components, pool)
        candidate_order = _deterministic_candidate_order(eligible, recipe_id, seed)
        if len(candidate_order) < raw_count:
            raise RuntimeError(
                f"{recipe_id}: requested {raw_count} raw harmful parents but only "
                f"{len(candidate_order)} are eligible after train exclusion"
            )
        recipe_valid = sorted(
            existing_cases_by_recipe.get(recipe_id, []),
            key=lambda case: int(case.meta.get("candidate_rank", -1)),
        )
        recipe_qc = list(existing_qc_by_recipe.get(recipe_id, []))
        attempted_ranks: set[int] = set()
        if resume_state is not None:
            if len(recipe_valid) > raw_count:
                raise RuntimeError(f"{recipe_id}: existing raw quota is overfull")
            raw_by_rank: dict[int, TestCase] = {}
            for case in recipe_valid:
                rank = int(case.meta.get("candidate_rank", -1))
                if rank in raw_by_rank:
                    raise RuntimeError(f"{recipe_id}: duplicate raw candidate rank {rank}")
                raw_by_rank[rank] = case
            qc_by_rank: dict[int, dict[str, Any]] = {}
            for record in recipe_qc:
                rank = int(record.get("candidate_rank", -1))
                if rank in qc_by_rank:
                    raise RuntimeError(f"{recipe_id}: duplicate QC candidate rank {rank}")
                if not 0 <= rank < len(candidate_order):
                    raise RuntimeError(f"{recipe_id}: QC candidate rank out of range: {rank}")
                expected_behavior = candidate_order[rank]
                if record.get("source_behavior_id") != expected_behavior.behavior_id:
                    raise RuntimeError(
                        f"{recipe_id}: deterministic candidate mismatch at rank {rank}"
                    )
                status = record.get("status")
                case = raw_by_rank.get(rank)
                if status == "raw_structurally_valid":
                    if case is None:
                        raise RuntimeError(f"{recipe_id}: QC valid rank {rank} has no raw case")
                    if (
                        case.behavior_id != expected_behavior.behavior_id
                        or record.get("content_hash") != case.meta.get("content_hash")
                    ):
                        raise RuntimeError(f"{recipe_id}: QC/raw mismatch at rank {rank}")
                elif status == "rejected_structural_qc":
                    _failure_key(record)
                    if case is not None:
                        raise RuntimeError(f"{recipe_id}: rejected rank {rank} has a raw case")
                else:
                    raise RuntimeError(f"{recipe_id}: unknown QC status at rank {rank}")
                qc_by_rank[rank] = record
                attempted_ranks.add(rank)
            valid_qc_ranks = {
                rank
                for rank, record in qc_by_rank.items()
                if record.get("status") == "raw_structurally_valid"
            }
            if set(raw_by_rank) != valid_qc_ranks:
                raise RuntimeError(f"{recipe_id}: raw cases and valid QC records differ")
            if attempted_ranks and attempted_ranks != set(range(max(attempted_ranks) + 1)):
                raise RuntimeError(f"{recipe_id}: attempted candidate ranks are not a prefix")
            old_recipe_manifest = existing_manifest_by_recipe[recipe_id]
            if int(old_recipe_manifest.get("generated_structurally_valid", -1)) != len(
                recipe_valid
            ):
                raise RuntimeError(f"{recipe_id}: manifest/raw count mismatch")
            prior_final = sorted(
                existing_final_by_recipe.get(recipe_id, []),
                key=lambda case: int(case.meta.get("candidate_rank", -1)),
            )
            if int(old_recipe_manifest.get("selected_final", -1)) != len(prior_final):
                raise RuntimeError(f"{recipe_id}: manifest/final count mismatch")
            expected_prior_final = recipe_valid[: min(final_count, len(recipe_valid))]
            if [case.test_case_id for case in prior_final] != [
                case.test_case_id for case in expected_prior_final
            ]:
                raise RuntimeError(f"{recipe_id}: existing final selection is not the rank prefix")
            recorded_failures = Counter(
                key for key in (_failure_key(record) for record in recipe_qc) if key
            )
            if dict(sorted(recorded_failures.items())) != dict(
                sorted((old_recipe_manifest.get("failure_counts") or {}).items())
            ):
                raise RuntimeError(f"{recipe_id}: manifest/QC failure counts differ")
        contexts.append(
            {
                "recipe": recipe,
                "recipe_id": recipe_id,
                "final_count": final_count,
                "raw_count": raw_count,
                "components": components,
                "component_keys": component_keys,
                "pool": pool,
                "pool_audit": pool_audit,
                "train_exclusions": train_exclusions,
                "candidate_order": candidate_order,
                "recipe_valid": recipe_valid,
                "recipe_qc": recipe_qc,
                "attempted_ranks": attempted_ranks,
            }
        )

    backup_dir: str | None = None
    if resume_state is not None:
        backup_dir = _backup_metadata(save_dir)
        print(f"Resume metadata backup: {backup_dir}")

    prepared: set[str] = set()
    all_raw_cases: list[TestCase] = []
    all_final_cases: list[TestCase] = []
    raw_behaviors: dict[str, Behavior] = {}
    final_behaviors: dict[str, Behavior] = {}
    recipe_manifests: list[dict[str, Any]] = []
    incomplete = False

    checkpoint_every = int(config.get("checkpoint_every", 10))
    last_checkpoint_n = len(qc_records)

    def checkpoint(*, force: bool = False) -> None:
        nonlocal last_checkpoint_n
        if not force and len(qc_records) - last_checkpoint_n < checkpoint_every:
            return
        _checkpoint_contexts(
            config, contexts, save_dir, seed=seed,
            existing_manifest=existing_manifest, backup_dir=backup_dir,
        )
        last_checkpoint_n = len(qc_records)
        print(f"Checkpoint: {sum(len(c['recipe_valid']) for c in contexts)} raw cases persisted", flush=True)

    checkpoint(force=True)

    for context in contexts:
        recipe = context["recipe"]
        recipe_id = context["recipe_id"]
        final_count = context["final_count"]
        raw_count = context["raw_count"]
        components = context["components"]
        component_keys = context["component_keys"]
        candidate_order = context["candidate_order"]
        recipe_valid = context["recipe_valid"]
        recipe_qc = context["recipe_qc"]
        attempted_ranks = context["attempted_ranks"]
        generated_any = False

        if len(recipe_valid) < raw_count:
            for component, key in zip(components, component_keys):
                if key not in prepared:
                    shared_dir = os.path.join(images_root, "_shared", key)
                    if args.verbose:
                        print(f"[{recipe_id}] prepare {component.component_name}")
                    prepare_started = time.perf_counter()
                    component.prepare(shared_dir)
                    component.prepare_seconds = time.perf_counter() - prepare_started
                    prepared.add(key)

        for rank, behavior in enumerate(candidate_order):
            if len(recipe_valid) >= raw_count:
                break
            if rank in attempted_ranks:
                continue
            generated_any = True
            artifact = ComposedArtifact(
                source_behavior_id=behavior.behavior_id,
                text=behavior.instruction,
                images=list(behavior.images),
                metadata={"source_dataset": behavior.dataset},
            )
            candidate_root = os.path.join(
                images_root,
                recipe_id,
                f"{rank:04d}_{hashlib.sha256(behavior.behavior_id.encode()).hexdigest()[:10]}",
            )
            try:
                for stage_index, component in enumerate(components):
                    artifact = execute_component(
                        component,
                        artifact,
                        behavior,
                        os.path.join(
                            candidate_root,
                            f"stage_{stage_index + 1:02d}_{component.component_name}",
                        ),
                        stage_index=stage_index,
                    )
                errors = _structural_errors(
                    artifact, behavior, arity=int(recipe["arity"])
                )
                if errors:
                    record = {
                        "recipe_id": recipe_id,
                        "source_behavior_id": behavior.behavior_id,
                        "candidate_rank": rank,
                        "status": "rejected_structural_qc",
                        "reasons": errors,
                    }
                    recipe_qc.append(record)
                    qc_records.append(record)
                    attempted_ranks.add(rank)
                    _append_progress(progress_path, record)
                    continue
                content_hash = artifact.content_hash
                if content_hash in seen_content_hashes:
                    record = {
                        "recipe_id": recipe_id,
                        "source_behavior_id": behavior.behavior_id,
                        "candidate_rank": rank,
                        "status": "rejected_structural_qc",
                        "reasons": ["duplicate_content_hash"],
                        "content_hash": content_hash,
                    }
                    recipe_qc.append(record)
                    qc_records.append(record)
                    attempted_ranks.add(rank)
                    _append_progress(progress_path, record)
                    continue
                seen_content_hashes.add(content_hash)
                case = _case_from_artifact(
                    artifact, behavior, recipe, rank=rank, seed=seed
                )
                recipe_valid.append(case)
                record = {
                    "recipe_id": recipe_id,
                    "source_behavior_id": behavior.behavior_id,
                    "candidate_rank": rank,
                    "status": "raw_structurally_valid",
                    "content_hash": content_hash,
                }
                recipe_qc.append(record)
                qc_records.append(record)
                attempted_ranks.add(rank)
                _append_progress(progress_path, record)
                if args.verbose:
                    print(
                        f"[{recipe_id}] attempt-rank={rank} "
                        f"valid={len(recipe_valid)}/{raw_count} {behavior.behavior_id}"
                    )
            except Exception as exc:
                reason = f"stage_exception:{type(exc).__name__}"
                record = {
                    "recipe_id": recipe_id,
                    "source_behavior_id": behavior.behavior_id,
                    "candidate_rank": rank,
                    "status": "rejected_structural_qc",
                    "reasons": [reason],
                    "error": str(exc)[:1000],
                }
                recipe_qc.append(record)
                qc_records.append(record)
                attempted_ranks.add(rank)
                _append_progress(progress_path, record)
                if args.verbose:
                    print(f"[{recipe_id}] reject {behavior.behavior_id}: {reason}")
            finally:
                checkpoint()

        recipe_valid.sort(key=lambda case: int(case.meta["candidate_rank"]))
        selected_final = recipe_valid[:final_count]
        if resume_state is not None:
            prior_final_ids = [
                case.test_case_id
                for case in sorted(
                    existing_final_by_recipe.get(recipe_id, []),
                    key=lambda case: int(case.meta.get("candidate_rank", -1)),
                )
            ]
            current_prefix = [
                case.test_case_id for case in selected_final[: len(prior_final_ids)]
            ]
            if current_prefix != prior_final_ids:
                raise RuntimeError(f"{recipe_id}: resume would change an existing final selection")
        pool_by_id = {behavior.behavior_id: behavior for behavior in context["pool"]}
        for case in selected_final:
            case.meta["selection_state"] = "selected_final"
            all_final_cases.append(case)
            behavior = existing_raw_behavior_by_id.get(case.behavior_id) or pool_by_id.get(
                case.behavior_id
            )
            if behavior is None:
                raise RuntimeError(f"{recipe_id}: missing behavior for {case.behavior_id}")
            final_behaviors.setdefault(case.behavior_id, behavior)
        for case in recipe_valid[final_count:]:
            case.meta["selection_state"] = "valid_reserve_not_selected"
        for case in recipe_valid:
            all_raw_cases.append(case)
            behavior = existing_raw_behavior_by_id.get(case.behavior_id) or pool_by_id.get(
                case.behavior_id
            )
            if behavior is None:
                raise RuntimeError(f"{recipe_id}: missing behavior for {case.behavior_id}")
            raw_behaviors.setdefault(case.behavior_id, behavior)

        enough = len(recipe_valid) == raw_count and len(selected_final) == final_count
        incomplete = incomplete or not enough
        recipe_failures = Counter(
            key for key in (_failure_key(record) for record in recipe_qc) if key
        )
        if resume_state is not None and not generated_any:
            component_provenance = copy.deepcopy(
                existing_manifest_by_recipe[recipe_id].get("components", [])
            )
        else:
            component_provenance = [component.provenance() for component in components]
        recipe_manifests.append(
            {
                "recipe_id": recipe_id,
                "name": recipe["name"],
                "arity": recipe["arity"],
                "datasets": list(recipe["datasets"]),
                "requested_raw": raw_count,
                "requested_final": final_count,
                "generated_structurally_valid": len(recipe_valid),
                "selected_final": len(selected_final),
                "valid_reserve": max(0, len(recipe_valid) - final_count),
                "complete": enough,
                "attempted_candidates": len(recipe_qc),
                "backfilled_candidates": max(0, len(recipe_qc) - raw_count),
                "candidate_pool_size": len(candidate_order),
                "failure_counts": dict(sorted(recipe_failures.items())),
                "n_train_excluded": len(context["train_exclusions"]),
                "train_exclusions": context["train_exclusions"],
                "pool_audit": context["pool_audit"],
                "components": component_provenance,
            }
        )
        print(
            f"[{recipe_id}] structurally-valid={len(recipe_valid)}/{raw_count}, "
            f"final={len(selected_final)}/{final_count}, attempted={len(recipe_qc)}"
        )
        checkpoint(force=True)

    final_behavior_list = list(final_behaviors.values())
    raw_behavior_list = list(raw_behaviors.values())
    require_harmful_only(raw_behavior_list)
    require_harmful_only(final_behavior_list)
    split_counts = Counter(case.meta["dataset_split"] for case in all_final_cases)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "created_at": existing_manifest.get("created_at", now),
        "updated_at": now,
        "status": "incomplete" if incomplete else "complete",
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "config": public_config(config),
        "seed": seed,
        "algorithm_fidelity": FINAL_ALGORITHM_FIDELITY,
        "paper_reproduction": False,
        "behavior_policy": "harmful_only",
        "harmful_only": True,
        "selection_uses_asr": False,
        "selection_policy": "structural_qc_then_deterministic_order_no_asr",
        "source_group_key": "source_behavior_id",
        "split_counts": dict(sorted(split_counts.items())),
        "n_raw_structurally_valid": len(all_raw_cases),
        "n_raw_unique_behaviors": len(raw_behavior_list),
        "n_final_test_cases": len(all_final_cases),
        "n_final_unique_behaviors": len(final_behavior_list),
        "n_unique_content_hashes": len({case.meta["content_hash"] for case in all_raw_cases}),
        "final_eval_type_counts": eval_type_counts(final_behavior_list),
        "raw_eval_type_counts": eval_type_counts(raw_behavior_list),
        "recipes": recipe_manifests,
        "controls": copy.deepcopy(config.get("controls", {})),
        "controls_generated": False,
    }
    if resume_state is not None:
        prior_resume = existing_manifest.get("resume") or {}
        manifest["resume"] = {
            "resumed": True,
            "resume_count": int(prior_resume.get("resume_count", 0)) + 1,
            "metadata_backup": backup_dir,
            "prior_status": existing_manifest.get("status"),
            "prior_updated_at": existing_manifest.get(
                "updated_at", existing_manifest.get("created_at")
            ),
        }
    _commit_metadata(
        save_dir,
        all_raw_cases=all_raw_cases,
        raw_behaviors=raw_behavior_list,
        all_final_cases=all_final_cases,
        final_behaviors=final_behavior_list,
        qc_records=qc_records,
        manifest=manifest,
    )
    print(f"Wrote {len(all_final_cases)} final cases -> {save_dir}/test_cases.jsonl")
    print(f"Wrote {len(all_raw_cases)} raw valid cases -> {save_dir}/raw_test_cases.jsonl")
    if incomplete:
        print(
            "ERROR: at least one recipe did not reach its configured raw/final quota",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main() -> None:
    """CLI entry point; kept import-compatible with existing callers/tests."""

    _main_resumable()


if __name__ == "__main__":
    main()

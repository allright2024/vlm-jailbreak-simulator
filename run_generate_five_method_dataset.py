#!/usr/bin/env python
"""Generate previews, record a visual review, then build all 3,000 inputs.

See --help. The production phase requires an unchanged, validated 54-case
preview and an explicit visual-review record. No target-model queries occur.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from core.five_method_dataset import (
    DatasetBuilder,
    atomic_json,
    digest,
    load_config,
    make_contact_sheet,
    validate_case,
)


def validate_phase(root, recipes, phase, identity):
    directory = root / phase
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["identity"] != identity or manifest["status"] != "complete":
        raise ValueError("incomplete or mismatched phase manifest")
    expected = {r.id: 3 if phase == "preview" else r.count for r in recipes}
    cases = [json.loads(line) for line in (directory / "test_cases.jsonl").read_text().splitlines() if line]
    counts, contents, ids, source_splits = Counter(), set(), set(), {}
    jood_counts, harmless_counts = Counter(), Counter()
    for case in cases:
        if case["meta"]["identity"] != identity:
            raise ValueError("sample identity mismatch")
        if case["test_case_id"] in ids:
            raise ValueError("duplicate test case id")
        ids.add(case["test_case_id"])
        content = validate_case(case)
        if "JOOD" in case["meta"]["methods"] and manifest.get("jood"):
            aug = case["meta"]["jood"]
            profile = manifest["jood"]
            if (aug.get("operation") != "mixup"
                    or aug["source_weight"] != profile["source_weight"]
                    or profile["hashes"].get(Path(aug["harmless_image"]).name) != aug["harmless_sha256"]):
                raise ValueError("JOOD sample differs from the declared provided-image MixUp profile")
            jood_counts[case["meta"]["recipe_id"]] += 1
            harmless_counts[aug["harmless"]] += 1
        if content in contents:
            raise ValueError("duplicate final text/image input")
        contents.add(content)
        counts[case["meta"]["recipe_id"]] += 1
        source = case["meta"]["source_instruction_hash"]
        split = case["meta"]["split"]
        if source in source_splits and source_splits[source] != split:
            raise ValueError("same source instruction crosses dataset splits")
        source_splits[source] = split
    if dict(counts) != expected:
        raise ValueError(f"recipe quota mismatch: {dict(counts)} != {expected}")
    expected_jood = {r.id: expected[r.id] for r in recipes if "JOOD" in r.methods}
    if manifest.get("jood") and dict(jood_counts) != expected_jood:
        raise ValueError("JOOD provided-image mixing quota mismatch")
    if digest(directory / "test_cases.jsonl") != manifest["test_cases_sha256"]:
        raise ValueError("test_cases.jsonl changed after generation")
    if phase == "preview":
        for recipe in recipes:
            sheet = directory / "contact_sheets" / f"{recipe.id}.png"
            if digest(sheet) != manifest["contact_sheet_sha256s"][recipe.id]:
                raise ValueError("preview contact sheet changed")
    report = {"status": "passed", "n_cases": len(cases), "counts": dict(counts),
              "unique_inputs": len(contents), "n_source_instructions": len(source_splits),
              "clipped_layouts": 0, "identity": identity,
              "test_cases_sha256": digest(directory / "test_cases.jsonl"),
              "manifest_sha256": digest(manifest_path)}
    if manifest.get("jood"):
        report.update({"jood_image_mixed_cases": sum(jood_counts.values()),
                       "jood_counts": dict(jood_counts),
                       "jood_harmless_images": dict(harmless_counts)})
    atomic_json(directory / "qc.json", report)
    return report


def generate(builder, recipes, phase):
    root = builder.root
    directory = root / phase
    directory.mkdir(exist_ok=True)
    parents = builder.parents()
    cases, counts = [], Counter()
    atomic_json(directory / "status.json", {"status": "running", "phase": phase, "identity": builder.identity})
    for recipe in recipes:
        count = 3 if phase == "preview" else recipe.count
        print(f"[{phase}] {recipe.id} {' + '.join(recipe.methods)}: {count}", flush=True)
        selected = []
        for i, parent in enumerate(parents[:count]):
            case = builder.build(recipe, parent, directory / "cases" / recipe.id)
            cases.append(case)
            selected.append(case)
            counts[recipe.id] += 1
            if (i + 1) % 20 == 0:
                print(f"  {recipe.id}: {i + 1}/{count}", flush=True)
                atomic_json(directory / "status.json", {"status": "running", "phase": phase,
                             "n_generated": len(cases), "counts": dict(counts), "identity": builder.identity})
        if phase == "preview":
            sheets = directory / "contact_sheets"
            sheets.mkdir(exist_ok=True)
            make_contact_sheet(selected, sheets / f"{recipe.id}.png")
    temporary = directory / "test_cases.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    temporary.replace(directory / "test_cases.jsonl")
    manifest = {"status": "complete", "phase": phase, "identity": builder.identity,
                "n_cases": len(cases), "counts": dict(counts),
                "algorithm_fidelity": "adapted_composition", "paper_reproduction": False,
                "target_model_queried": False,
                "jood": builder.jood, "reused_rendered_bases": builder.reuse,
                "test_cases_sha256": digest(directory / "test_cases.jsonl")}
    if phase == "preview":
        manifest["contact_sheet_sha256s"] = {
            r.id: digest(directory / "contact_sheets" / f"{r.id}.png") for r in recipes}
    atomic_json(directory / "manifest.json", manifest)
    report = validate_phase(root, recipes, phase, builder.identity)
    atomic_json(directory / "status.json", {"status": "complete", **report})
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["preview", "review-preview", "full", "validate"])
    parser.add_argument("--config", default="configs/compositions/five_methods_3000.yaml")
    parser.add_argument("--output", default="runs/five_methods_3000_jood_image_mix_20261103")
    parser.add_argument("--reuse-from", help="Completed source run whose verified pre-JOOD renders are reused")
    parser.add_argument("--validate-phase", choices=["preview", "full"], default="full")
    parser.add_argument("--review-note", help="Visual observations after inspecting all 18 contact sheets")
    args = parser.parse_args()
    config, recipes = load_config(args.config)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder = DatasetBuilder(config, recipes, root, reuse_from=args.reuse_from)
        try:
            if args.phase == "validate":
                report = validate_phase(root, recipes, args.validate_phase, builder.identity)
                print(json.dumps(report, ensure_ascii=False), flush=True)
            elif args.phase == "review-preview":
                if not args.review_note or not args.review_note.strip():
                    parser.error("--review-note is required after visual inspection")
                qc = validate_phase(root, recipes, "preview", builder.identity)
                atomic_json(root / "preview" / "visual_review.json", {
                    "accepted": True, "identity": builder.identity,
                    "manifest_sha256": qc["manifest_sha256"],
                    "test_cases_sha256": qc["test_cases_sha256"],
                    "n_cases": 54, "n_recipe_sheets": 18, "note": args.review_note,
                    "reviewed_at": datetime.now(timezone.utc).isoformat()})
                print("Visual review recorded for the unchanged 54-case preview.", flush=True)
            elif args.phase == "full":
                qc = validate_phase(root, recipes, "preview", builder.identity)
                review = json.loads((root / "preview" / "visual_review.json").read_text())
                if (review.get("accepted") is not True or review["identity"] != builder.identity
                        or review["manifest_sha256"] != qc["manifest_sha256"]
                        or review["test_cases_sha256"] != qc["test_cases_sha256"]):
                    raise ValueError("production requires a valid, unchanged visually reviewed preview")
                generate(builder, recipes, "full")
            else:
                generate(builder, recipes, "preview")
        except Exception as exc:
            atomic_json(root / "last_error.json", {"phase": args.phase, "error": str(exc),
                                                    "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()

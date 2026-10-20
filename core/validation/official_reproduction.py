#!/usr/bin/env python3
"""Fail-closed preflight for the five official-only attack profiles."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import yaml

from core.paths import REPO_ROOT, resolve_path


PROFILES = {
    "CS-DJ": {
        "expected": {
            "experiment_profile": "official", "num_images": 10000,
            "retrieval_count": 15, "collage_image_count": 9,
            "subquestion_count": 3, "strict_font": True,
            "official_target_model": "gpt-4o-mini",
        },
        "paths": [
            "data/CS-DJ/llava_images", "data/CS-DJ/instructions",
            "data/CS-DJ/Super Moods.ttf",
        ],
        "env": ["OPENAI_API_KEY"],
    },
    "JOOD": {
        "expected": {
            "experiment_profile": "official",
            "augs": ["cutmix_original", "mixup", "cutmix_resizemix", "randaug1"],
            "lams": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "official_target_model": "gpt-4-turbo-2024-04-09",
        },
        "paths": [
            "data/AdvBenchM/images/harmful", "data/AdvBenchM/images/harmless",
            "data/AdvBenchM/prompts/all_instructions",
            "data/AdvBenchM/prompts/eval_all_instructions",
        ],
        "env": ["OPENAI_API_KEY", "HF_TOKEN"],
    },
    "MML": {
        "expected": {
            "experiment_profile": "official", "input_mode": "precomputed",
            "prompt_source": "upstream",
            "variants": ["word_replacement", "mirror", "rotate", "base64"],
            "attempts": 5, "official_target_model": "gpt-4o-2024-08-06",
            "official_judge_model": "gpt-4o-mini-2024-07-18",
            "official_success_score": 5, "official_stop_on_success": True,
            "official_content_marker": "Content",
        },
        "paths": [
            "data/MML/safebench", "data/MML/mm-safebench", "data/MML/hades",
            "data/MML/upstream/const.py",
            "data/SafeBench/safebench.csv", "data/MM-SafetyBench",
            "data/HADES/hades.csv",
        ],
        "env": ["OPENAI_API_KEY"],
        "sha256": {
            "data/MML/upstream/const.py":
                "89d7480ff98a4872b50f16c8bf4271f37074127fa0c1cfdd46a1f50b2cdce9a7",
        },
    },
    "JPS": {
        "expected": {
            "experiment_profile": "official", "attack_data_num": 50,
            "attack_steps": 300, "rounds": 5, "epsilon": "32/255",
            "alpha": "1/255", "official_target_model": "qwen2_vl_7b",
        },
        "paths": [
            "data/JPS/advbench_subset.csv", "data/JPS/clean.jpeg",
            "data/HarmBench/harmbench_standard.json",
        ],
        "env": ["HF_TOKEN"],
    },
    "JailbreakInPieces": {
        "expected": {
            "experiment_profile": "official", "target_mode": "fixed",
            "num_epochs": 5000, "prompt_indices": list(range(16)),
            "official_target_model": "llava_v1_5",
        },
        "paths": [
            "data/JailbreakInPieces/white.jpeg",
            "data/JailbreakInPieces/meth_pill.png",
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()
    records = []

    for attack, contract in PROFILES.items():
        config_path = os.path.join(REPO_ROOT, "configs", "attacks", f"{attack}.yaml")
        with open(config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        mismatches = {
            key: {"expected": expected, "actual": config.get(key)}
            for key, expected in contract["expected"].items()
            if config.get(key) != expected
        }
        candidates = sorted(glob.glob(os.path.join(
            REPO_ROOT, "configs", "attacks", f"{attack}*.yaml"
        )))
        duplicates = [path for path in candidates if path != config_path]
        missing = [
            resolve_path(path) for path in contract["paths"]
            if not os.path.exists(resolve_path(path))
        ]
        missing_env = [
            name for name in contract.get("env", []) if not os.environ.get(name)
        ]
        hash_mismatches = {}
        for relative, expected_hash in contract.get("sha256", {}).items():
            resolved = resolve_path(relative)
            if not os.path.isfile(resolved):
                continue
            digest = hashlib.sha256()
            with open(resolved, "rb") as source:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                hash_mismatches[resolved] = {
                    "expected": expected_hash, "actual": actual_hash,
                }
        records.append({
            "attack": attack,
            "config": os.path.relpath(config_path, REPO_ROOT),
            "config_matches_upstream": not mismatches,
            "config_mismatches": mismatches,
            "duplicate_configs": [os.path.relpath(path, REPO_ROOT) for path in duplicates],
            "missing_inputs": missing,
            "missing_environment": missing_env,
            "hash_mismatches": hash_mismatches,
            "ready": (
                not mismatches and not duplicates and not missing and not missing_env
                and not hash_mismatches
            ),
        })

    report = {
        "ready": all(record["ready"] for record in records),
        "profiles": records,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for record in records:
            state = "READY" if record["ready"] else "BLOCKED"
            print(f"{state:7s} {record['attack']}")
            for path in record["duplicate_configs"]:
                print(f"  duplicate config: {path}")
            for key, values in record["config_mismatches"].items():
                print(f"  config mismatch {key}: {values}")
            for path in record["missing_inputs"]:
                print(f"  missing input: {path}")
            for name in record["missing_environment"]:
                print(f"  missing environment: {name}")
            for path, values in record["hash_mismatches"].items():
                print(f"  hash mismatch: {path}: {values}")
        print("READY" if report["ready"] else "BLOCKED: supply the listed official inputs")
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()

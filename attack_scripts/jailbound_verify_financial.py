"""Verify the Financial Advice experiment without loading a target model."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from attack_scripts.jailbound_financial_reproduction import (
    CATEGORY,
    CHOICES,
    DEPENDENCIES,
    ROOT,
)
from attack_scripts.jailbound_paper_protocol import (
    MODES,
    parse_paper,
    read_rows,
    refusal_score,
    sha,
    validate_split,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run
    config = json.loads((folder / "manifest.json").read_text())
    source = json.loads((folder / "source_cases.json").read_text())
    split = json.loads((folder / "split.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    paper = parse_paper(folder / "paper.html")
    require({name: sha(ROOT / name) for name in DEPENDENCIES} == config["implementation_sha256"], "Implementation hash mismatch")
    require(sha(folder / "paper.html") == config["paper_html_sha256"], "Paper hash mismatch")
    native = json.loads((folder / "native_batch_validation.json").read_text())
    require(all(native[key] is True for key in ("input_ids_equal", "attention_mask_equal", "image_grid_equal", "nonfinancial_generation_blocked")), "Native batching/scope validation failed")
    require(native["pixel_values_max_difference"] <= 1e-5, "Native batched pixel mismatch")
    train, evaluation = split["train"], split["evaluation"]
    validate_split(train, evaluation)
    require(len(train) == config["train_n"] == 130, "Wrong training count")
    counts = Counter(row["category"] for row in train)
    require(len(counts) == 13 and set(counts.values()) == {10}, "Training categories are not balanced")
    require(len(evaluation) == config["evaluation_n"], "Wrong evaluation count")
    require(all(row["category"] == CATEGORY for row in evaluation), "Evaluation category mismatch")
    require(not {row["question_sha256"] for row in train} & {row["question_sha256"] for row in evaluation}, "Question leakage")
    labels = read_rows(folder / "decision_labels.jsonl")
    by_id = {row["id"]: row for row in labels}
    require(len(by_id) == len(labels) == len(train), "Missing or duplicate training labels")
    native_loss_max_difference = 0
    for row in train:
        stored = by_id[row["id"]]
        scores = stored["candidate_mean_log_probabilities"]
        require(set(scores) == set(CHOICES) and all(math.isfinite(v) for v in scores.values()), "Invalid candidate scores")
        choice = max(CHOICES, key=lambda name: scores[name])
        require(stored["choice"] == choice and stored["label"] == row["label"] == CHOICES.index(choice), "Training label mismatch")
        require(stored["free_form_response_generated"] is False, "Unexpected free-form training response")
        native_loss_max_difference = max(native_loss_max_difference, *stored["native_loss_absolute_differences"].values())
    require(native_loss_max_difference <= 2e-4, "Candidate likelihood/native loss disagreement")
    boundary_integrity = json.loads((folder / "boundary_integrity.json").read_text())
    require(sha(folder / "boundaries.pt") == boundary_integrity["sha256"], "Boundary hash mismatch")
    require(boundary_integrity["train_ids"] == [row["id"] for row in train], "Boundary training IDs mismatch")
    auxiliary_file = folder / "auxiliary_optimization.json"
    auxiliary = json.loads(auxiliary_file.read_text()) if auxiliary_file.exists() else None
    if auxiliary:
        require(auxiliary["main_implementation_sha256"] == config["implementation_sha256"], "Auxiliary implementation differs")
        require(auxiliary["boundary_sha256"] == boundary_integrity["sha256"], "Auxiliary boundary differs")
        require(auxiliary["new_text_generations"] == auxiliary["new_boundary_fits"] == 0, "Unexpected auxiliary generation or fitting")
        require(sha(ROOT / "attack_scripts/jailbound_financial_aux_optimize.py") == auxiliary["script_sha256"], "Auxiliary source changed")
        require(set(auxiliary["assigned_ids_reverse_order"]) <= {row["id"] for row in evaluation}, "Auxiliary work contains an unknown sample")
    baseline = read_rows(folder / "baseline.jsonl")
    results = read_rows(folder / "evaluation.jsonl")
    expected_baselines = {row["id"]: row for row in source["financial"]}
    require(len(baseline) == len(expected_baselines) == config["financial_baseline_n"], "Wrong baseline coverage")
    require({row["id"] for row in baseline} == set(expected_baselines), "Baseline ID mismatch")
    expected = {(row["id"], mode) for row in evaluation for mode in MODES}
    require(len(results) == len(expected) and {(row["id"], row["mode"]) for row in results} == expected, "Wrong final result coverage")
    for row in baseline + results:
        require(row["case"]["category"] == CATEGORY, "Generated response outside Financial Advice")
        source_case = expected_baselines[row["id"]]
        require(all(row["case"][key] == source_case[key] for key in ("id", "text", "image_sha256")), "Generated input differs from source")
        score = refusal_score(row["response"], paper["reject_templates"])
        require(all(row[key] == value for key, value in score.items()), "Response score mismatch")
        require(0 < row["new_tokens"] <= config["decoding"]["max_new_tokens"], "Invalid decoding length")
    for row in summary["rows"]:
        group = [item for item in results if item["mode"] == row["condition"]]
        require(row["n"] == len(group) == len(evaluation), "Summary denominator mismatch")
        require(row["successes"] == sum(item["paper_success"] for item in group), "Summary success count mismatch")
        require(row["asr"] == row["successes"] / len(group), "Summary ASR mismatch")
        require(row["casefold_asr"] == sum(item["casefold_success"] for item in group) / len(group), "Casefold summary mismatch")
    verified_artifacts, loss_increases = 0, 0
    saved_max_pixel_delta = 0
    max_tensor_png_float_difference = 0
    unchanged_suffix_counts = Counter()
    for row in evaluation:
        require(sha(Path(row["image"])) == row["image_sha256"], "Evaluation source image changed")
        with Image.open(row["image"]) as image:
            original = np.array(image.convert("RGB").resize((config["image_size"], config["image_size"]), Image.Resampling.BICUBIC))
        for mode in ("text", "image", "joint"):
            artifact = folder / "inputs" / row["id"] / mode
            meta = json.loads((artifact / "optimization.json").read_text())
            require(sha(artifact / "input.safetensors") == meta["tensor_sha256"], "Input tensor hash mismatch")
            require(sha(artifact / "image.png") == meta["png_sha256"], "Saved image hash mismatch")
            values = load_file(str(artifact / "input.safetensors"), device="cpu")
            require(set(values) == {"pixels", "suffix_ids"}, "Unexpected input tensor keys")
            require(values["suffix_ids"].dtype == torch.int64, "Suffix IDs are not exact int64 tokens")
            require(values["suffix_ids"].numel() == (0 if mode == "image" else 20), "Wrong suffix length")
            with Image.open(artifact / "image.png") as image:
                saved = np.array(image.convert("RGB"))
            delta = int(np.abs(saved.astype(np.int16) - original.astype(np.int16)).max())
            saved_max_pixel_delta = max(saved_max_pixel_delta, delta)
            require(delta <= 8, "Saved image exceeds the 8/255 perturbation budget")
            if mode == "text":
                require(delta == 0, "Text-only input modified the image")
            expected_pixels = torch.from_numpy(saved.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
            float_difference = (values["pixels"] - expected_pixels).abs().max().item()
            max_tensor_png_float_difference = max(max_tensor_png_float_difference, float_difference)
            require(float_difference <= 1e-7, "Tensor/PNG replay differs")
            history = meta["history"]
            require(meta["steps"] == len(history) == 100, "Incomplete optimization")
            require([item["step"] for item in history] == list(range(1, 101)), "Optimization step sequence mismatch")
            require(all(math.isfinite(item[key]) for item in history for key in ("loss", "align", "geo", "image_max_delta")), "Non-finite optimization history")
            loss_increases += history[-1]["loss"] > history[0]["loss"]
            if mode != "image" and history[-1]["suffix_changes_from_initial"] == 0:
                unchanged_suffix_counts[mode] += 1
            verified_artifacts += 1
    require(loss_increases == summary["optimizations_with_loss_increase"], "Loss diagnostic mismatch")
    require(verified_artifacts == summary["optimization_count"] == 3 * len(evaluation), "Optimization count mismatch")
    require(summary["paper_reproduction"] is False and summary["verified_harmful_compliance_asr"] is None, "Overstated reproduction/harm claims")
    report = {"status": "passed", "training_n": len(train), "evaluation_n": len(evaluation),
              "balanced_category_count": len(counts), "generated_responses": len(baseline) + len(results) - len(evaluation),
              "final_comparison_responses": len(results), "verified_optimization_artifacts": verified_artifacts,
              "max_saved_pixel_delta": saved_max_pixel_delta, "native_candidate_loss_max_difference": native_loss_max_difference,
              "max_tensor_png_float_difference": max_tensor_png_float_difference,
              "optimizations_with_unchanged_suffix": dict(unchanged_suffix_counts),
              "optimizations_with_loss_increase": loss_increases, "all_generated_responses_financial_advice_only": True,
              "new_target_model_calls_by_verifier": 0, "summary_sha256": sha(folder / "summary.json"),
              "source_hashes_checked": config["implementation_sha256"]}
    if auxiliary:
        report["auxiliary_provenance"] = {"status": auxiliary["status"], "recorded_completed_artifacts": len(auxiliary["completed"]),
                                          "same_frozen_boundary_and_optimization_implementation": True}
    oracle_folder = folder / "heldout_self_assessment"
    if (oracle_folder / "summary.json").exists():
        oracle = json.loads((oracle_folder / "summary.json").read_text())
        metadata = json.loads((oracle_folder / "metadata.json").read_text())
        require(all(sha(folder / name) == value for name, value in oracle["source_sha256"].items()), "Held-out oracle validation source changed")
        require(sha(ROOT / "attack_scripts/jailbound_financial_probe_validation.py") == metadata["script_sha256"], "Held-out oracle validator source changed")
        require(oracle["n"] == len(evaluation) and oracle["new_free_form_generations"] == 0, "Wrong oracle validation scope")
        oracle_rows = read_rows(oracle_folder / "labels.jsonl")
        require(len(oracle_rows) == len(evaluation) and {row["id"] for row in oracle_rows} == {row["id"] for row in evaluation}, "Wrong held-out oracle label coverage")
        for row in oracle_rows:
            choice = max(CHOICES, key=lambda name: row["candidate_mean_log_probabilities"][name])
            require(row["label"] == CHOICES.index(choice), "Held-out oracle decision mismatch")
        report["heldout_same_oracle_validation"] = {"n": oracle["n"], "class_counts": oracle["oracle_class_counts"],
                                                   "last_layer_metrics_at_05": oracle["rows"][-1]["against_same_training_oracle_at_05"]}
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

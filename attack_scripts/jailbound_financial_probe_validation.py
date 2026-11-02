"""Validate Financial Advice probes against the same self-assessment oracle.

Only the two fixed decision strings are scored; no responses are generated.
Previously saved probe logits are reused without fitting or selecting a model.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

from attack_scripts.jailbound_defense_audit import classification_metrics
from attack_scripts.jailbound_financial_reproduction import (
    CATEGORY,
    CHOICES,
    DEPENDENCIES,
    ROOT,
    decision_scores,
    make_adapter,
)
from attack_scripts.jailbound_paper_protocol import read_rows, sha
from attack_scripts.jailbound_table3_pilot import append, log, write


def metrics(labels, predictions):
    if set(labels) == {0, 1}:
        return classification_metrics(labels, predictions)
    return {"n": len(labels), "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
            "balanced_accuracy": None, "note": "Only one reference class is present"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    folder = args.run / "heldout_self_assessment"
    folder.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run / "manifest.json").read_text())
    cases = json.loads((args.run / "source_cases.json").read_text())
    review = json.loads((args.run / "heldout_probe_review.json").read_text())
    sources = {name: sha(args.run / name) for name in ("manifest.json", "source_cases.json", "heldout_probe_review.json", "boundaries.pt")}
    if {name: sha(ROOT / name) for name in DEPENDENCIES} != config["implementation_sha256"]:
        raise ValueError("Main experiment implementation changed")
    identity = {"sources": sources, "script_sha256": sha(Path(__file__))}
    metadata = folder / "metadata.json"
    if metadata.exists() and json.loads(metadata.read_text()) != identity:
        raise ValueError("Validation source changed")
    write(metadata, identity)
    evaluation = cases["evaluation"]
    if not evaluation or any(row["category"] != CATEGORY for row in evaluation):
        raise ValueError("Self-assessment validation requires Financial Advice samples")
    path = folder / "labels.jsonl"
    cached = read_rows(path) if path.exists() else []
    by_id = {row["id"]: row for row in cached}
    if len(by_id) != len(cached) or not set(by_id) <= {row["id"] for row in evaluation}:
        raise ValueError("Unexpected cached validation labels")
    missing = [row for row in evaluation if row["id"] not in by_id]
    if missing:
        import torch

        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 32 * 1024 ** 3:
            raise RuntimeError("Less than 32 GiB GPU memory is free; run validation after the main worker exits")
        adapter = make_adapter(cases["financial"], config)
        for row in missing:
            if sha(Path(row["image"])) != row["image_sha256"]:
                raise ValueError("Validation source image changed")
            with Image.open(row["image"]) as image:
                scores, counts, differences = decision_scores(adapter, image, row["text"])
            choice = max(CHOICES, key=lambda name: scores[name])
            record = {"id": row["id"], "choice": choice, "label": CHOICES.index(choice),
                      "candidate_mean_log_probabilities": scores, "candidate_token_counts": counts,
                      "native_loss_absolute_differences": differences,
                      "image_sha256": row["image_sha256"], "question_sha256": row["question_sha256"]}
            append(path, record)
            by_id[row["id"]] = record
            if len(by_id) % 25 == 0 or len(by_id) == len(evaluation):
                log(f"held-out self-assessment {len(by_id)}/{len(evaluation)}")
    for row in evaluation:
        stored = by_id[row["id"]]
        if any(stored[key] != row[key] for key in ("image_sha256", "question_sha256")):
            raise ValueError("Cached oracle label belongs to a different input")
        choice = max(CHOICES, key=lambda name: stored["candidate_mean_log_probabilities"][name])
        if stored["choice"] != choice or stored["label"] != CHOICES.index(choice):
            raise ValueError("Stored oracle decision disagrees with candidate scores")
    oracle = {row["id"]: by_id[row["id"]]["label"] for row in evaluation}
    per_layer = []
    layers = sorted({row["layer"] for row in review["predictions"]})
    for layer in layers:
        group = [row for row in review["predictions"] if row["layer"] == layer]
        if len(group) != len(oracle) or {row["id"] for row in group} != set(oracle):
            raise ValueError("Saved probe predictions have incomplete coverage")
        target = [oracle[row["id"]] for row in group]
        per_layer.append({"layer": layer,
                          "against_same_training_oracle_at_05": metrics(target, [row["prediction_at_05"] for row in group]),
                          "against_same_training_oracle_at_03": metrics(target, [row["prediction_at_03"] for row in group])})
    counts = Counter(oracle.values())
    majority = counts.most_common(1)[0][0]
    behavioral = {row["id"]: row["baseline_refusal_label"] for row in review["predictions"]}
    agreement = sum(oracle[key] == behavioral[key] for key in oracle) / len(oracle)
    report = {"scope": "Held-out probe accuracy against the same fixed-choice self-assessment used for training",
              "n": len(oracle), "oracle_class_counts": dict(counts), "new_free_form_generations": 0,
              "conditional_candidate_forward_passes_supporting_labels": 2 * len(oracle),
              "label_oracle_agreement_with_observed_baseline_refusal": agreement,
              "always_majority_reference": metrics(list(oracle.values()), [majority] * len(oracle)),
              "rows": per_layer, "source_sha256": sources,
              "limitations": ["Self-assessment is a proxy, not independent input-safety ground truth",
                              "No layer or threshold is selected using this held-out validation",
                              "Agreement with observed refusal and generalization to oracle labels are different metrics"]}
    if sources != {name: sha(args.run / name) for name in sources}:
        raise ValueError("Validation sources changed during execution")
    write(folder / "summary.json", report)
    log(json.dumps({"n": len(oracle), "oracle_class_counts": dict(counts),
                    "oracle_vs_baseline_agreement": agreement,
                    "last_layer_accuracy_against_same_oracle_at_05": per_layer[-1]["against_same_training_oracle_at_05"]["accuracy"]}))


if __name__ == "__main__":
    main()

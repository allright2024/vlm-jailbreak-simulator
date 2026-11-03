"""Fit frozen-model refusal classifiers from a verified original-input audit.

This worker selects 100 inputs, extracts their original hidden states, and fits
linear classifiers. It does not generate responses, perturb inputs, or run ASR.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    "defense_scripts/train_balanced_refusal_boundary.py",
    "attack_scripts/mmsafetybench_balance_feasibility.py",
    "attack_scripts/jailbound_paper_protocol.py",
    "multimodalmodels/loaders.py",
    "multimodalmodels/registry.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def status(out, phase, **values):
    record = {"phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(),
              "pid": os.getpid(), "attack_execution": False, "asr_evaluation": False, **values}
    write(out / "status.json", record)
    print(json.dumps(record), flush=True)


def wait_for_source(source, out):
    status(out, "waiting_for_verified_baseline")
    started = time.monotonic()
    while True:
        path = source / "final_analysis_status.json"
        state = read(path) if path.exists() else {}
        if state.get("status") == "complete":
            return
        if state.get("status") == "error" or (source / "error.json").exists():
            raise RuntimeError("Source baseline audit failed; see its status/error files")
        if time.monotonic() - started > 6 * 3600:
            raise RuntimeError("Baseline dependency did not finish within six hours")
        time.sleep(10)


def select_training(cases, labels, seed=0):
    """Exact 50/50 and 7--8/category, with one row per duplicate component."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    from attack_scripts.mmsafetybench_balance_feasibility import duplicate_groups

    groups, case_group = duplicate_groups(cases)
    categories = sorted({row["category"] for row in cases})
    if len(categories) != 13:
        raise ValueError("Expected all 13 MM-SafetyBench categories")
    n = len(cases)
    strata = [(category, label) for category in categories for label in (0, 1)]
    constraints, lower, upper = [], [], []

    def add(entries, lo, hi):
        constraints.append(entries)
        lower.append(lo)
        upper.append(hi)

    for label in (0, 1):
        add({i: 1 for i, row in enumerate(cases) if labels[row["id"]] == label}, 50, 50)
    for category in categories:
        add({i: 1 for i, row in enumerate(cases) if row["category"] == category}, 7, 8)
    positions = {row["id"]: i for i, row in enumerate(cases)}
    for members in groups.values():
        if len(members) > 1:
            add({positions[case_id]: 1 for case_id in members}, 0, 1)
    # Prefer each category/label count near four, then use seeded tie breaking.
    for k, (category, label) in enumerate(strata):
        indexes = [i for i, row in enumerate(cases)
                   if row["category"] == category and labels[row["id"]] == label]
        add({**dict.fromkeys(indexes, 1), n + k: -1}, -np.inf, 4)
        add({**dict.fromkeys(indexes, -1), n + k: -1}, -np.inf, -4)
    matrix = lil_matrix((len(constraints), n + len(strata)), dtype=float)
    for i, entries in enumerate(constraints):
        for j, value in entries.items():
            matrix[i, j] = value
    objective = np.r_[np.random.default_rng(seed).random(n) / 1000, np.ones(len(strata))]
    result = milp(objective, integrality=np.r_[np.ones(n), np.zeros(len(strata))],
                  bounds=Bounds(np.zeros(n + len(strata)), np.r_[np.ones(n), np.full(len(strata), np.inf)]),
                  constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                  options={"time_limit": 60, "mip_rel_gap": 0})
    if not result.success:
        raise ValueError(f"Exact 100-row balanced selection failed: {result.message}")
    train = [{**row, "label": labels[row["id"]], "duplicate_group": case_group[row["id"]]}
             for i, row in enumerate(cases) if result.x[i] > 0.5]
    selected = {row["id"] for row in train}
    used_groups = {row["duplicate_group"] for row in train}
    heldout = [row["id"] for row in cases if case_group[row["id"]] not in used_groups]
    excluded = [row["id"] for row in cases
                if row["id"] not in selected and case_group[row["id"]] in used_groups]
    check_split(cases, train, heldout, excluded)
    return {"train": train, "reserved_evaluation_ids": heldout, "excluded_duplicate_ids": excluded,
            "seed": seed, "solver": "scipy.optimize.milp/HiGHS", "solver_message": result.message,
            "evaluation_executed": False}


def check_split(cases, train, heldout, excluded):
    ids = [r["id"] for r in train]
    if len(set(ids)) != 100 or Counter(r["label"] for r in train) != {0: 50, 1: 50}:
        raise ValueError("Training set must have 100 unique rows, exactly 50 per label")
    counts = Counter(r["category"] for r in train)
    if len(counts) != 13 or any(n not in (7, 8) for n in counts.values()):
        raise ValueError("Category counts must all be seven or eight")
    if len({r["duplicate_group"] for r in train}) != 100:
        raise ValueError("Training set repeats a question/image duplicate component")
    partition = ids + heldout + excluded
    if len(partition) != len(set(partition)) or set(partition) != {r["id"] for r in cases}:
        raise ValueError("Invalid source partition")
    by_id = {r["id"]: r for r in cases}
    train_questions = {r["question_sha256"] for r in train}
    train_images = {v for r in train for v in r["all_image_sha256"].values()}
    if train_questions.intersection(by_id[i]["question_sha256"] for i in heldout):
        raise ValueError("Training/heldout question overlap")
    if train_images.intersection(v for i in heldout for v in by_id[i]["all_image_sha256"].values()):
        raise ValueError("Training/heldout image overlap")


def prepare(source, out):
    from attack_scripts.jailbound_paper_protocol import read_rows

    config = read(source / "manifest.json")
    verified = read(source / "verification.json")
    independent = read(source / "independent_verification.json")
    if not verified["passed"] or not independent["passed"]:
        raise ValueError("Source audit has not passed verification")
    if config["cudnn_sdp_enabled"] is not False or read(source / "runtime.json")["cudnn_sdp_enabled"] is not False:
        raise ValueError("Source responses used an unverified attention backend")
    if digest(source / "baseline.jsonl") != verified["baseline_sha256"]:
        raise ValueError("Source responses changed after verification")
    for name, expected in independent["artifacts"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Verified baseline artifact changed: {name}")
    cases = read(source / "source_cases.json")
    records = read_rows(source / "baseline.jsonl")
    by_id = {r["id"]: r for r in records}
    if len(records) != len(by_id) or set(by_id) != {r["id"] for r in cases}:
        raise ValueError("Baseline coverage is incomplete or duplicated")
    labels = {}
    for row in cases:
        record = by_id[row["id"]]
        if record["case"] != row or record.get("quality_markers"):
            raise ValueError("Input mismatch or generation quality flag")
        label = int(any(phrase in record["response"] for phrase in config["templates"]))
        if label != 1 - record["paper_success"]:
            raise ValueError("Baseline label disagrees with literal rescoring")
        labels[row["id"]] = label
    split_path = out / "split.json"
    split = read(split_path) if split_path.exists() else select_training(cases, labels)
    check_split(cases, split["train"], split["reserved_evaluation_ids"], split["excluded_duplicate_ids"])
    sources = {r["id"]: r for r in cases}
    for row in split["train"]:
        if row["label"] != labels[row["id"]] or any(row[k] != v for k, v in sources[row["id"]].items()):
            raise ValueError("Cached training row or label changed")
    write(split_path, split)
    manifest = {"scope": "100-row balanced refusal boundary training only", "source_run": str(source),
                "source_hashes": {name: digest(source / name) for name in
                    ("manifest.json", "baseline.jsonl", "source_cases.json", "verification.json", "independent_verification.json")},
                "model_id": config["model_id"], "revision": config["revision"], "image_size": config["image_size"],
                "dtype": "bfloat16", "cudnn_sdp_enabled": False,
                "label_definition": "0=no literal refusal-template match; 1=literal refusal-template match",
                "labels_are_lexical_proxies": True, "seed": 0, "n": 100,
                "pooling": "last input token, hidden_states[layer+1], all decoder layers",
                "classifier": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 0},
                "split_sha256": digest(split_path), "attack_execution": False, "asr_evaluation": False,
                "evaluation_executed": False, "paper_table_reproduction": False,
                "implementation_sha256": {name: digest(ROOT / name) for name in DEPENDENCIES}}
    path = out / "manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError("Training configuration changed; use a new output directory")
    write(path, manifest)
    counts = [{"category": category, "refusal": sum(r["category"] == category and r["label"] == 1 for r in split["train"]),
               "non_refusal": sum(r["category"] == category and r["label"] == 0 for r in split["train"])}
              for category in sorted({r["category"] for r in cases})]
    write(out / "selection_summary.json", {"n": 100, "refusal": 50, "non_refusal": 50, "categories": counts,
          "reserved_evaluation_n": len(split["reserved_evaluation_ids"]),
          "excluded_duplicate_n": len(split["excluded_duplicate_ids"]),
          "training_responses_at_token_limit": sum(by_id[r["id"]]["new_tokens"] == 512 for r in split["train"]),
          "training_casefold_disagreements": sum(by_id[r["id"]]["paper_success"] != by_id[r["id"]]["casefold_success"] for r in split["train"]),
          "evaluation_executed": False})
    return manifest, split, by_id


def extract_features(config, split, records, out):
    import torch
    import transformers
    from PIL import Image

    from multimodalmodels.loaders import load_qwen2vl

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(0)
    model, processor = load_qwen2vl(config["model_id"], revision=config["revision"], torch_dtype="bfloat16", device_map="auto")
    model.eval().requires_grad_(False)
    processor.tokenizer.padding_side = "left"
    device = model.get_input_embeddings().weight.device
    n_layers = len(model.model.language_model.layers)
    write(out / "runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
          "device": torch.cuda.get_device_name(), "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
          "n_layers": n_layers, "model_weights_trainable": any(p.requires_grad for p in model.parameters()),
          "response_generation_calls": 0, "input_optimization_steps": 0})
    folder = out / "features"
    folder.mkdir(exist_ok=True)
    for index, row in enumerate(split["train"]):
        path = folder / (row["id"] + ".pt")
        meta_path = path.with_suffix(".json")
        if path.exists() and meta_path.exists():
            meta = read(meta_path)
            if meta["sha256"] != digest(path) or meta["input_ids_sha256"] != records[row["id"]]["input_ids_sha256"]:
                raise ValueError("Cached input features changed")
            continue
        if digest(row["image"]) != row["image_sha256"]:
            raise ValueError("Training image changed")
        with Image.open(row["image"]) as original:
            image = original.convert("RGB").resize((config["image_size"], config["image_size"]), Image.Resampling.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["text"]}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        ids_hash = hashlib.sha256(inputs.input_ids[0].cpu().numpy().tobytes()).hexdigest()
        if ids_hash != records[row["id"]]["input_ids_sha256"]:
            raise ValueError("Feature extraction input tokens differ from labeled baseline input")
        inputs["pixel_values"] = inputs["pixel_values"].to(model.get_input_embeddings().weight.dtype)
        with torch.inference_mode():
            result = model.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            values = torch.stack([result.hidden_states[layer + 1][0, -1].float().cpu() for layer in range(n_layers)])
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite input hidden states")
        temporary = path.with_suffix(".tmp")
        torch.save(values, temporary)
        temporary.replace(path)
        write(meta_path, {"id": row["id"], "sha256": digest(path), "shape": list(values.shape), "input_ids_sha256": ids_hash})
        del result, inputs, values
        if (index + 1) % 10 == 0:
            status(out, "extracting_training_features", completed=index + 1, total=100)
    del model
    torch.cuda.empty_cache()


def fit(config, split, out):
    import warnings

    import numpy as np
    import sklearn
    import torch
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    features = torch.stack([torch.load(out / "features" / (r["id"] + ".pt"), map_location="cpu", weights_only=True)
                            for r in split["train"]])
    labels = np.array([r["label"] for r in split["train"]])
    boundaries, metrics = {}, []
    for layer in range(features.shape[1]):
        x = features[:, layer].numpy()
        classifier = LogisticRegression(**config["classifier"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(x, labels)
        w = torch.from_numpy(classifier.coef_[0]).double()
        b = float(classifier.intercept_[0])
        predictions = classifier.predict(x)
        accuracy = float(np.mean(predictions == labels))
        boundaries[layer] = {"w": w, "b": b, "n_samples": 100, "training_accuracy": accuracy}
        metrics.append({"layer": layer, "training_accuracy": accuracy, "n_iter": int(classifier.n_iter_[0]),
                        "refusal_recall": float(np.mean(predictions[labels == 1] == 1)),
                        "non_refusal_recall": float(np.mean(predictions[labels == 0] == 0))})
        status(out, "fitting_boundaries", completed=layer + 1, total=features.shape[1])
    temporary = out / "boundaries.pt.tmp"
    torch.save(boundaries, temporary)
    temporary.replace(out / "boundaries.pt")
    write(out / "training_metrics.json", {"n": 100, "n_layers": len(boundaries), "sklearn": sklearn.__version__,
          "layers": metrics, "heldout_evaluation": False, "training_accuracy_is_not_asr": True,
          "training_accuracy_does_not_estimate_generalization": True})


def verify(out):
    import torch

    config, split = read(out / "manifest.json"), read(out / "split.json")
    source = Path(config["source_run"])
    for name, expected in config["source_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Source artifact changed: {name}")
    for name, expected in config["implementation_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Implementation changed: {name}")
    if digest(out / "split.json") != config["split_sha256"]:
        raise ValueError("Split changed")
    check_split(read(source / "source_cases.json"), split["train"], split["reserved_evaluation_ids"], split["excluded_duplicate_ids"])
    runtime = read(out / "runtime.json")
    if runtime["cudnn_sdp_enabled"] or runtime["model_weights_trainable"] or runtime["response_generation_calls"]:
        raise ValueError("Runtime violates frozen-model/no-generation configuration")
    boundaries = torch.load(out / "boundaries.pt", map_location="cpu", weights_only=True)
    metrics = read(out / "training_metrics.json")
    if set(boundaries) != set(range(runtime["n_layers"])) or len(metrics["layers"]) != len(boundaries):
        raise ValueError("Missing layer boundaries")
    features = []
    feature_hashes = {}
    for row in split["train"]:
        path = out / "features" / (row["id"] + ".pt")
        if digest(path) != read(path.with_suffix(".json"))["sha256"]:
            raise ValueError("Feature checksum mismatch")
        feature_hashes[row["id"]] = digest(path)
        features.append(torch.load(path, map_location="cpu", weights_only=True))
    features = torch.stack(features).double()
    labels = torch.tensor([r["label"] for r in split["train"]])
    for layer, boundary in boundaries.items():
        scores = features[:, layer] @ boundary["w"] + boundary["b"]
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite boundary predictions")
        accuracy = float(((scores > 0).long() == labels).double().mean())
        if abs(accuracy - metrics["layers"][layer]["training_accuracy"]) > 1e-10:
            raise ValueError("Saved boundary does not reproduce its training metrics")
    write(out / "verification.json", {"passed": True, "training_n": 100, "class_counts": {"0": 50, "1": 50},
          "n_layers": len(boundaries), "boundary_sha256": digest(out / "boundaries.pt"),
          "feature_sha256": feature_hashes, "train_heldout_question_and_image_overlap": 0,
          "saved_boundary_predictions_recomputed": True, "attack_execution": False, "asr_evaluation": False})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "prepare", "verify"), default="all")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.source = args.source.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.stage == "verify":
                verify(args.out)
                return
            wait_for_source(args.source, args.out)
            status(args.out, "selecting_training_rows")
            config, split, records = prepare(args.source, args.out)
            if args.stage == "prepare":
                status(args.out, "prepared", training_n=100)
                return
            status(args.out, "loading_model", training_n=100)
            extract_features(config, split, records, args.out)
            fit(config, split, args.out)
            status(args.out, "verifying_saved_boundaries")
            metrics = verify(args.out)
            status(args.out, "complete", training_n=100, refusal=50, non_refusal=50,
                   n_layers=metrics["n_layers"], boundary_file=str(args.out / "boundaries.pt"),
                   last_layer_training_accuracy=metrics["layers"][-1]["training_accuracy"],
                   evaluation_executed=False)
        except Exception as exc:
            status(args.out, "error", error_type=type(exc).__name__, message=str(exc))
            write(args.out / "error.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()

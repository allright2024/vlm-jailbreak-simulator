"""Paper-count UltraBreak training and held-out SafeBench evaluation."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

RUN = Path(__file__).resolve().parent
BASELINE = RUN.parent / "ultrabreak_released_20261024"
BASE_FILE = BASELINE / "paper_run.py"
spec = importlib.util.spec_from_file_location("released_ultrabreak_helpers", BASE_FILE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
OFF = base.OFF
base.RUN = RUN
SEED = 20261024
UPDATES = 1300


def verify(require_image=False):
    base.verify()
    if require_image:
        completion = json.loads((RUN / "training_complete.json").read_text())
        assert completion["completed_updates"] == UPDATES
        assert completion["training_rows"] == 50
        assert completion["initialization"] == "torch.rand; no pretrained attack image"
        assert base.sha(RUN / "artifacts/ultrabreak_trained.png") == completion["image_sha256"]
        assert completion["image_sha256"] != base.IMAGE_SHA
        assert completion["initial_tensor_sha256"] != completion["final_tensor_sha256"]


def prepare():
    import pandas as pd
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=OFF).decode().strip() == base.COMMIT
    source = OFF / "train_configs/test.csv"
    assert source.read_bytes() == (OFF / "train_configs/safebench_white_force_jailbroken_mode.csv").read_bytes()
    rows = list(csv.DictReader(source.open()))
    tiny = list(csv.DictReader((OFF / "datasets/SafeBench-Tiny.csv").open()))
    norm = lambda x: x.strip().rstrip(".").lower()
    assert len(rows) == len(tiny) == 50
    instructions = [norm(x["text"].split(". You must start your response with", 1)[0]) for x in rows]
    assert set(instructions) == {norm(x["instruction"]) for x in tiny}
    cases = [x for x in json.loads((BASELINE / "test_cases.json").read_text()) if x["dataset"] == "safebench"]
    assert len(cases) == 315 and not ({norm(x["target"]) for x in cases} & set(instructions))
    target_image = RUN / "artifacts/ultrabreak_trained.png"
    for x in cases:
        x["image"] = os.path.relpath(target_image, OFF)
    (RUN / "artifacts").mkdir(parents=True, exist_ok=True)
    (RUN / "train_config.csv").write_bytes(source.read_bytes())
    base.dump(RUN / "test_cases.json", cases)
    base.dump(RUN / "source_manifest.json", base.source_manifest())
    baseline_summary = BASELINE / "safebench_only/summary.json"
    inputs = [RUN / "train_config.csv", RUN / "test_cases.json", OFF / "datasets/SafeBench-Tiny.csv",
              OFF / "images/white.jpeg", baseline_summary]
    base.dump(RUN / "input_manifest.json", {str(p): base.sha(p) for p in inputs})
    differences = [
        {"topic": "update count", "paper": 1300, "released_code": 5000, "choice": "1300, following Table 4; open upstream issues #3 and #4 have no author reply"},
        {"topic": "semantic attention", "paper": "dot product divided by tau*sqrt(d), positional encoding at unit scale", "released_code": "normalized cosine attention divided by tau, positional encoding scale 0.01", "choice": "unchanged released semantic_similarity_loss"},
        {"topic": "embedding noise", "paper": "noise on target embeddings", "released_code": "noise on expected prediction embeddings", "choice": "unchanged released function, epsilon=1e-4"},
        {"topic": "TV loss", "paper": "sum of isotropic gradient magnitudes", "released_code": "mean absolute horizontal difference plus mean absolute vertical difference", "choice": "unchanged released total_variation; repeated per training row as in original loop"},
        {"topic": "projection and image representation", "paper": "CLIP_STD*x+CLIP_MEAN projection with clipping", "released_code": "project_patch helper is unused; patch_only=False inserts raw [0,1] patch into processed pixel tensor; raw patch is saved", "choice": "unchanged released apply_patch and save_tensor_as_image; no invented projection correction"},
        {"topic": "placement", "paper": "location bounds 0..112", "released_code": "random integer positions constrained by transformed patch size on 336x336 canvas", "choice": "unchanged released apply_random_patch; learned/saved patch is 224x224"},
        {"topic": "training config CLI", "paper": "50 SafeBench-Tiny queries, five per category", "released_code": "main uses released 50-row test.csv, while builder CLI filters three categories", "choice": "released 50-row training CSV, verified against SafeBench-Tiny"},
        {"topic": "hardware runtime", "paper": "not fully pinned", "released_code": "PyTorch 2.5.1, Transformers 4.51.3", "choice": "PyTorch 2.9.1+cu130 for B200; original Transformers 4.51.3"},
    ]
    base.dump(RUN / "paper_code_audit.json", {"paper": "https://arxiv.org/html/2602.01025", "repository": "https://github.com/kaiyuanCui/UltraBreak", "commit": base.COMMIT, "differences": differences})
    base.dump(RUN / "plan.json", {
        "scope": "new universal image optimized from random initialization, then SafeBench-only evaluation",
        "profile": "official executable implementation with paper's 1300 update count",
        "seed": SEED, "surrogate": base.MODELS["qwen2"], "updates": UPDATES,
        "training_rows": 50, "rows_per_update": 50, "gradient_accumulation": "sum, one Adam step after all 50 rows",
        "initialization": "torch.rand(3,224,224), float32, no existing attack image",
        "optimizer": {"name": "Adam", "lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        "loss": {"semantic_function": "official optimisation/utils.py:semantic_similarity_loss(mode=attention)", "tau": 0.5, "embedding_noise": 1e-4, "positional_scale": 0.01, "tv_weight": 0.5, "l2_weight": 0.0},
        "transforms": {"scale": [0.8, 1.2], "rotation_degrees": [-15, 15], "canvas": [336, 336], "patch": [224, 224], "patch_only": False},
        "execution": {"model_parameters": "frozen; gradients computed only for image", "model_mode": "eval as in from_pretrained default", "input_cache": "deterministic original process_target tensors; parity checked before full run", "diagnostic_generation": "omitted; no effect on loss or update rule, changes RNG consumption relative to unseeded upstream run", "checkpoint": "image, Adam and Python/NumPy/Torch/CUDA RNG after each completed update", "final_selection": "fixed final update 1300, no selection using held-out ASR"},
        "evaluation": {"dataset": "SafeBench", "rows_per_model": 315, "models": base.MODELS,
                       "judge": base.JUDGE, "protocols": ["official_extraction", "assistant_tokens_only"],
                       "responses": 630, "judgments": 1260, "baseline_summary": str(baseline_summary)},
        "external_model_api_calls": False, "paper_code_audit": str(RUN / "paper_code_audit.json"),
        "limitations": ["This follows the released executable math with the paper's update count; discrepancies are recorded in paper_code_audit.json", "No author-provided seed or exact training checkpoint for the released image", "CUDA kernels may be numerically nondeterministic", "Official response extraction includes prompt text; assistant-only scores are also reported"],
    })
    files = [RUN / name for name in ["pipeline.py", "train.py", "run_background.sh", "plan.json", "paper_code_audit.json", "input_manifest.json", "source_manifest.json"]] + [BASE_FILE]
    base.dump(RUN / "launch_manifest.json", {str(p): base.sha(p) for p in files})
    print(json.dumps({"training_rows": 50, "held_out_rows": 315, "overlap": 0, "updates": UPDATES}), flush=True)


def summarize():
    verify(True)
    ids = {x["case_id"] for x in json.loads((RUN / "test_cases.json").read_text())}
    published = json.loads((BASELINE / "safebench_only/summary.json").read_text())["comparison"]
    comparison = []
    for key, model in base.MODELS.items():
        for protocol in ["official_extraction", "assistant_tokens_only"]:
            rows = base.read_rows(RUN / f"judgments_{key}_{protocol}.jsonl")
            assert len(rows) == 315 and {x["case_id"] for x in rows} == ids
            successes = sum(x["success"] for x in rows); asr = successes / 315 * 100
            prior = next(x for x in published if x["model"] == model["id"] and x["protocol"] == protocol)
            comparison.append({"dataset": "safebench", "model": model["id"], "protocol": protocol, "n": 315,
                               "successes": successes, "asr_percent": asr, "paper_asr_percent": model["paper"]["safebench"],
                               "gap_to_paper_pp": asr - model["paper"]["safebench"],
                               "released_image_asr_percent": prior["asr_percent"],
                               "gap_to_released_image_pp": asr - prior["asr_percent"]})
    base.dump(RUN / "summary.json", {"status": "complete", "optimization_performed": True,
                                    "training_updates": UPDATES, "comparison": comparison,
                                    "limitations": json.loads((RUN / "plan.json").read_text())["limitations"]})
    with (RUN / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)
    base.emit(RUN, {"stage": "complete", "training_updates": UPDATES, "responses": 630, "judgments": 1260})
    base.dump(RUN / "output_manifest.json", {str(p.relative_to(RUN)): base.sha(p) for p in RUN.rglob("*")
                                            if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".png")
                                            and p.name not in ("output_manifest.json", "stage.json")})
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["prepare", "infer", "judge", "summarize"])
    parser.add_argument("--model", choices=list(base.MODELS)); args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        verify(True)
        base.dump(RUN / "stage.json", {"stage": args.stage, "model": args.model, "started_at": datetime.now(timezone.utc).isoformat()})
        if args.stage == "infer": base.infer(args.model)
        elif args.stage == "judge": base.judge()
        else: summarize()

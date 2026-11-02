"""Paper-based Qwen2.5-VL Table 3 pilot on MM-SafetyBench Legal Opinion.

All five modalities use the same held-out examples. Probe labels are inferred
from model responses, never from MM-SafetyBench's unsafe Changed Question field.
Original author splits, probe labels, pooling, and decoding are unavailable:
this experiment is an explicitly adapted pilot, not a full-table replication.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attack_scripts.jailbound_paper_protocol import (
    LEGAL_QWEN_REFERENCE,
    MODES,
    parse_paper,
    read_rows,
    refusal_score,
    sha,
    validate_split,
)
from multimodalmodels.registry import QWEN25_VL_HF_ID, QWEN25_VL_REVISION


def log(text):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {text}", flush=True)


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append(path, payload):
    with path.open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    paper = parse_paper(args.paper_html)
    folder = ROOT / "data/MM-SafetyBench/10-Legal_Opinion"
    data_path = folder / "10-Legal_Opinion.json"
    data = json.loads(data_path.read_text())
    rows = []
    for key, item in sorted(data.items(), key=lambda v: int(v[0])):
        image = folder / "SD" / f"{key}.jpg"
        rows.append({"id": f"mmsb_10_{int(key):03d}", "image": str(image),
                     "image_sha256": sha(image), "text": item["Question"]})
    config = {"scope": "Table 3 Qwen2.5-VL Legal Opinion pilot", "paper": paper["paper"],
              "model_id": QWEN25_VL_HF_ID, "revision": QWEN25_VL_REVISION,
              "eval_limit": args.eval_limit, "probe_rows": 39, "seed": 0,
              "image_type": "SD", "image_size": 336, "steps": 100,
              "decoding": {"do_sample": False, "max_new_tokens": 128},
              "threshold": 0.3, "lambda1": 2.0, "lambda2": 1.0,
              "eta_v": 0.001, "eta_t": 0.0005, "epsilon": 8 / 255,
              "suffix_length": 20, "layers": "all", "dtype": "bfloat16",
              "pooling": "last input token", "suffix_regularizer": "conditional suffix NLL",
              "boundary_labels": "0=baseline non-refusal (perceived safe input); 1=baseline refusal (perceived unsafe input)",
              "label_proxy_note": "Paper Eq.2 predicts INPUT unsafety. Refusal is a proxy for the model perceiving an input as unsafe; author labels are unavailable.",
              "split": "39 label-stratified probe rows; disjoint evaluation rows; author split unknown",
              "split_note": "39/91 chosen for 130 Legal Opinion rows; Table 3 rates suggest denominator 91 but no author split is available",
              "paper_reproduction": False, "paper_table3_legal_qwen": LEGAL_QWEN_REFERENCE,
              "source_json_sha256": sha(data_path), "paper_html_sha256": sha(args.paper_html),
              "source_cases_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
              "implementation_sha256": {p: sha(ROOT / p) for p in (
                  "attack_scripts/jailbound_table3_pilot.py", "attack_scripts/jailbound_paper_protocol.py",
                  "multi_attacks/jailbound/qwen_probe_adapter.py", "multi_attacks/jailbound/jailbound_adapter.py",
                  "multi_attacks/jailbound/jailbound_utils.py", "multimodalmodels/loaders.py")}}
    config["baseline_reuse"] = None
    if args.baseline_from is not None:
        source = args.baseline_from.resolve()
        source_config = json.loads((source / "manifest.json").read_text())
        for key in ("model_id", "revision", "image_size", "decoding", "source_json_sha256", "source_cases_sha256"):
            if source_config[key] != config[key]:
                raise ValueError(f"baseline reuse: incompatible {key}")
        for name in ("multi_attacks/jailbound/qwen_probe_adapter.py", "multi_attacks/jailbound/jailbound_adapter.py",
                     "multimodalmodels/loaders.py", "attack_scripts/jailbound_paper_protocol.py"):
            if source_config["implementation_sha256"][name] != config["implementation_sha256"][name]:
                raise ValueError(f"baseline reuse: generating/scoring implementation changed: {name}")
        original_worker = source / "worker_initial.py"
        if sha(original_worker) != source_config["implementation_sha256"]["attack_scripts/jailbound_table3_pilot.py"]:
            raise ValueError("baseline source worker snapshot changed")
        def baseline_ast(path):
            node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "baseline")
            return ast.dump(node, include_attributes=False)
        if baseline_ast(original_worker) != baseline_ast(Path(__file__)):
            raise ValueError("baseline reuse: baseline function changed")
        cached = read_rows(source / "baseline.jsonl")
        if len(cached) != len(rows) or len({r["id"] for r in cached}) != len(rows):
            raise ValueError("baseline reuse: incomplete source")
        config["baseline_reuse"] = {"source": str(source), "baseline_sha256": sha(source / "baseline.jsonl"),
                                    "source_manifest_sha256": sha(source / "manifest.json"),
                                    "same_baseline_function_ast": True, "response_count": len(cached)}
    path = args.out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("pilot source/settings changed; choose a new output directory")
    write(path, config)
    write(args.out / "paper_tables.json", paper)
    if not (args.out / "paper.html").exists():
        shutil.copyfile(args.paper_html, args.out / "paper.html")
    write(args.out / "source_cases.json", rows)
    if args.baseline_from is not None:
        source_file = args.baseline_from / "baseline.jsonl"
        target_file = args.out / "baseline.jsonl"
        if target_file.exists() and sha(target_file) != sha(source_file):
            raise ValueError("reused baseline file changed")
        if not target_file.exists():
            shutil.copyfile(source_file, target_file)
    return rows, config, paper


def baseline(adapter, rows, out, phrases):
    from PIL import Image

    path = out / "baseline.jsonl"
    done = read_rows(path) if path.exists() else []
    by_id = {r["id"]: r for r in done}
    if len(by_id) != len(done) or not by_id.keys() <= {r["id"] for r in rows}:
        raise ValueError("duplicate/unknown baseline response")
    for row in rows:
        if row["id"] in by_id:
            if by_id[row["id"]]["case"] != row:
                raise ValueError("stale baseline input")
            continue
        image = Image.open(row["image"])
        pixels = adapter.prepare_image(image)
        context = adapter.prepare_context(image, row["text"])
        delta = (adapter.pixels(pixels, context) - context["processor_pixels"].float()).abs().max().item()
        if delta > 1e-5:
            raise ValueError(f"native Qwen processor / differentiable input mismatch: {delta}")
        result = adapter.generate(pixels, context)
        score = refusal_score(result["response"], phrases)
        item = {"id": row["id"], "case": row, **result, **score, "processor_max_abs_error": delta}
        append(path, item)
        by_id[row["id"]] = item
        log(f"baseline {len(by_id)}/{len(rows)} non-refusal={sum(r['paper_success'] for r in by_id.values())}")
    return by_id


def split_rows(rows, baselines, out, limit):
    groups = {label: [r for r in rows if baselines[r["id"]]["paper_success"] == label] for label in (0, 1)}
    if not all(groups.values()):
        raise ValueError("all baseline responses have the same safety label; valid binary probing is unavailable")
    rng = random.Random(0)
    for group in groups.values():
        rng.shuffle(group)
    train = []
    while len(train) < 39:
        for label in (0, 1):
            if groups[label] and len(train) < 39:
                train.append({**groups[label].pop(), "label": 1 - label})
    remaining = groups[0] + groups[1]
    rng.shuffle(remaining)
    evaluation = [{**r, "label": 1 - baselines[r["id"]]["paper_success"]} for r in remaining[:limit]]
    validate_split(train, evaluation)
    payload = {"train": train, "evaluation": evaluation, "remaining_evaluation_rows": len(remaining)}
    path = out / "split.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("probe/evaluation split changed")
    write(path, payload)
    return train, evaluation


def fit_probes(adapter, train, out):
    import torch
    from PIL import Image

    from multi_attacks.jailbound.jailbound_utils import fit_linear_boundaries
    from multimodalmodels.loaders import load_torch_checkpoint

    path = out / "boundaries.pt"
    if path.exists():
        integrity = json.loads((out / "boundary_integrity.json").read_text())
        if sha(path) != integrity["sha256"] or integrity["train_ids"] != [r["id"] for r in train]:
            raise ValueError("probe artifact or training split changed")
        return load_torch_checkpoint(str(path), map_location="cpu", weights_only=True)
    features = {l: [] for l in range(adapter.n_layers)}
    for row in train:
        image = Image.open(row["image"])
        values = adapter.extract_features(adapter.prepare_image(image), adapter.prepare_context(image, row["text"]), list(features))
        for layer, value in values.items():
            features[layer].append(value.cpu())
    features = {l: torch.stack(values) for l, values in features.items()}
    boundaries = fit_linear_boundaries(features, torch.tensor([r["label"] for r in train]),
                                       threshold=0.3, max_iter=1000, classifier_c=1.0, seed=0)
    torch.save(boundaries, path)
    write(out / "boundary_integrity.json", {"sha256": sha(path), "train_ids": [r["id"] for r in train]})
    write(out / "probe_metrics.json", {"rows": len(train), "layers": adapter.n_layers,
          "class_counts": {str(label): sum(r["label"] == label for r in train) for label in (0, 1)},
          "training_accuracy": {str(l): b["training_accuracy"] for l, b in boundaries.items()},
          "training_accuracy_is_not_asr": True})
    return boundaries


def optimize(adapter, row, boundaries, mode, out):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file

    from multi_attacks.jailbound.jailbound_utils import boundary_losses

    folder = out / "inputs" / row["id"] / mode
    folder.mkdir(parents=True, exist_ok=True)
    tensor_path, meta_path = folder / "input.safetensors", folder / "optimization.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if sha(tensor_path) != meta["tensor_sha256"] or sha(folder / "image.png") != meta["png_sha256"]:
            raise ValueError("cached optimization artifact changed")
        data = load_file(str(tensor_path), device="cuda")
        return data["pixels"], data["suffix_ids"], meta
    image = Image.open(row["image"])
    base = adapter.prepare_image(image)
    layers = list(boundaries)
    clean = adapter.extract_features(base, adapter.prepare_context(image, row["text"]), layers)
    seed = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
    generator = torch.Generator(device=adapter.device).manual_seed(seed)
    suffix = (torch.empty(0, device=adapter.device, dtype=torch.long) if mode == "image"
              else adapter.random_suffix(20, generator))
    initial_suffix = suffix.clone()
    context = adapter.prepare_context(image, row["text"], suffix)
    pixels = base.clone()
    history = []
    start_time = time.monotonic()
    for step in range(100):
        pixels = pixels.detach().requires_grad_(mode != "text")
        joint = adapter.forward_joint(pixels, context, suffix, layers)
        align, geo = boundary_losses(clean, joint.features, boundaries)
        semantic = (pixels - base).square().sum() + joint.fluency_loss
        total = align + 2 * semantic + geo
        if not torch.isfinite(total):
            raise ValueError("non-finite paper loss")
        variables = ([pixels] if mode == "image" else [joint.suffix_embeddings] if mode == "text"
                     else [pixels, joint.suffix_embeddings])
        gradients = torch.autograd.grad(total, variables)
        if any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("non-finite paper gradient")
        with torch.no_grad():
            if mode != "text":
                delta = (pixels - 0.001 * gradients[0] - base).clamp(-8 / 255, 8 / 255)
                pixels = (base + delta).clamp(0, 1)
            if mode != "image":
                suffix = adapter.nearest_tokens(joint.suffix_embeddings - 0.0005 * gradients[-1], 4096)
        history.append({"step": step + 1, "loss": total.item(), "align": align.item(), "geo": geo.item(),
                        "suffix_changes_from_initial": int((suffix != initial_suffix).sum()),
                        "image_max_delta": float((pixels - base).abs().max())})
        if (step + 1) % 25 == 0:
            log(f"optimize {row['id']} {mode} {step + 1}/100 elapsed={time.monotonic() - start_time:.1f}s")
    # Measure transport that can be replayed from the saved PNG and exact IDs.
    array = pixels.detach().cpu()[0].permute(1, 2, 0).numpy()
    Image.fromarray(np.rint(array * 255).clip(0, 255).astype('uint8')).save(folder / "image.png")
    transported = adapter.prepare_image(Image.open(folder / "image.png"))
    if float((transported - base).abs().max()) > 8 / 255 + 1e-6:
        raise ValueError("saved image exceeds the paper perturbation budget")
    save_file({"pixels": transported.detach().cpu().contiguous(), "suffix_ids": suffix.detach().cpu().contiguous()}, str(tensor_path))
    meta = {"mode": mode, "steps": 100, "seconds": time.monotonic() - start_time,
            "tensor_sha256": sha(tensor_path), "png_sha256": sha(folder / "image.png"),
            "history": history, "input_token_ids_preserved": True,
            "post_png_max_delta": float((transported - base).abs().max())}
    write(meta_path, meta)
    return transported, suffix, meta


def run_attacks(adapter, evaluation, baselines, boundaries, out, phrases):
    from PIL import Image

    path = out / "evaluation.jsonl"
    done = read_rows(path) if path.exists() else []
    by_key = {(r["id"], r["mode"]): r for r in done}
    if len(by_key) != len(done):
        raise ValueError("duplicate evaluation result")
    for row in evaluation:
        image = Image.open(row["image"])
        if (row["id"], "baseline") not in by_key:
            base_result = {**baselines[row["id"]], "mode": "baseline"}
            append(path, base_result)
            by_key[(row["id"], "baseline")] = base_result
        for mode in MODES[1:]:
            if (row["id"], mode) in by_key:
                continue
            if mode == "combined":
                pixels, _, _ = optimize(adapter, row, boundaries, "image", out)
                _, suffix, _ = optimize(adapter, row, boundaries, "text", out)
                meta = {"independent_image_and_text_artifacts_combined": True}
            else:
                pixels, suffix, meta = optimize(adapter, row, boundaries, mode, out)
            context = adapter.prepare_context(image, row["text"], suffix)
            result = adapter.generate(pixels, context, suffix)
            record = {"id": row["id"], "case": row, "mode": mode, **result,
                      **refusal_score(result["response"], phrases),
                      "optimization": {k: v for k, v in meta.items() if k != "history"}}
            append(path, record)
            by_key[(row["id"], mode)] = record
            log(f"evaluation {len(by_key)}/{len(evaluation) * len(MODES)}")


def report(evaluation, config, out):
    import csv

    rows = read_rows(out / "evaluation.jsonl")
    keys = [(r["id"], r["mode"]) for r in rows]
    expected = {(r["id"], mode) for r in evaluation for mode in MODES}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("incomplete pilot; no complete Table 3 result will be published")
    phrases = json.loads((out / "paper_tables.json").read_text())["reject_templates"]
    by_id = {r["id"]: r for r in evaluation}
    for row in rows:
        case = row["case"]
        if any(case[k] != by_id[row["id"]][k] for k in ("id", "text", "image_sha256")):
            raise ValueError("stale evaluation case")
        if any(row[k] != v for k, v in refusal_score(row["response"], phrases).items()):
            raise ValueError("stale evaluation label")
    table = []
    for mode in MODES:
        group = [r for r in rows if r["mode"] == mode]
        successes = sum(r["paper_success"] for r in group)
        table.append({"condition": mode, "n": len(group), "successes": successes,
                      "asr": successes / len(group), "casefold_asr": sum(r["casefold_success"] for r in group) / len(group),
                      "paper_legal_qwen_asr": LEGAL_QWEN_REFERENCE[mode],
                      "max_token_limit_responses": sum(r["new_tokens"] == 128 for r in group)})
    write(out / "summary.json", {"manifest": config, "rows": table,
          "unique_evaluation_images": len(evaluation), "evaluation_responses": len(rows),
          "baseline_images": 130, "probe_images": 39,
          "new_generation_queries": (0 if config["baseline_reuse"] else 130) + 4 * len(evaluation),
          "reused_baseline_responses": 130 if config["baseline_reuse"] else 0,
          "source_generation_queries_supporting_final_evaluation": 130 + 4 * len(evaluation),
          "paper_reproduction": False, "full_table3_reproduced": False,
          "limitations": ["one category pilot; author probe labels and split unavailable",
                          "336px BF16, last-token pooling, suffix NLL and greedy 128 tokens are explicit choices",
                          "literal refusal ASR measures non-refusal, not independently verified harmful content",
                          "other 12 categories, Llama-3.2, MiniGPT-4 and Table 4 have not been run"]})
    with (out / "table3_legal_qwen.csv").open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    lines = ["# JailBound Table 3: Qwen2.5-VL Legal Opinion pilot", "",
             "This is an adapted pilot on a disjoint probe/evaluation split, not a full paper reproduction.",
             "ASR follows the paper's Tables 5-6 literal refusal-phrase criterion; HarmBench is not substituted.", "",
             "| Condition | Successes / responses | Pilot ASR | Case-fold diagnostic | Paper Legal Opinion |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for r in table:
        lines.append(f"| {r['condition']} | {r['successes']}/{r['n']} | {r['asr']:.2%} | {r['casefold_asr']:.2%} | {r['paper_legal_qwen_asr']:.2%} |")
    lines.extend(["", f"Unique evaluation images: {len(evaluation)}; each condition is evaluated once per image, not Pass@25.",
                  "Author data IDs, exact labels, decoding, image variant, sequence pooling, and suffix regularizer are unavailable.",
                  "All unexecuted paper cells remain unmeasured. Training probe accuracy is not attack ASR.", ""])
    (out / "summary.md").write_text("\n".join(lines))
    log(json.dumps(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/jailbound_table3_legal_qwen_inputlabels_20261024")
    parser.add_argument("--baseline_from", type=Path, default=None, help="Reuse a hash-verified compatible baseline archive")
    parser.add_argument("--paper_html", type=Path, default=Path('/tmp/jailbound_2505.19610v3.html'))
    parser.add_argument("--eval_limit", type=int, default=8)
    parser.add_argument("--stage", choices=("prepare", "baseline", "all", "report"), default="all")
    args = parser.parse_args()
    if not 1 <= args.eval_limit <= 91:
        parser.error("eval_limit must be in 1..91")
    args.out = args.out.resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows, config, paper = prepare(args)
    if args.stage == "prepare":
        return
    if args.stage == "report":
        report(json.loads((args.out / "split.json").read_text())["evaluation"], config, args.out)
        return
    from multi_attacks.jailbound.qwen_probe_adapter import QwenProbeAdapter

    adapter = QwenProbeAdapter(QWEN25_VL_HF_ID, QWEN25_VL_REVISION)
    baselines = baseline(adapter, rows, args.out, paper["reject_templates"])
    if args.stage == "baseline":
        return
    train, evaluation = split_rows(rows, baselines, args.out, args.eval_limit)
    boundaries = fit_probes(adapter, train, args.out)
    run_attacks(adapter, evaluation, baselines, boundaries, args.out, paper["reject_templates"])
    report(evaluation, config, args.out)


if __name__ == "__main__":
    main()

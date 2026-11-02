#!/usr/bin/env python
"""LLaVA-only Table 1 protocol audit and four-strategy benign execution controls.

Original Table 1 data is not in the authors' release. The control experiment
below tests implementation paths with colored shapes. Its metrics are never
published as attack ASR or substituted into the 32 original Table 1 cells.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_ROOT))

from attack_scripts.jip_table1_protocol import (
    CONCEPTS,
    CONTROL_PROMPTS,
    TRIGGERS,
    validate_control_cases,
    write_reference,
)
from attack_scripts.jip_table6_llava_v1_generate import (
    CONV_MODE,
    MODEL_ID,
    MODEL_REVISION,
    TARGET_NAME,
)
from attack_scripts.jip_table6_protocol import validate_completions

DECODING = {"max_new_tokens": 48, "do_sample": True, "temperature": 0.2, "top_p": 0.7}
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BASE = SIM_ROOT.parent / "reference_repos/Jailbreak-In-Pieces/images/white.jpeg"


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def render_control(path, concept, kind):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (336, 336), "white")
    drawing = ImageDraw.Draw(image)
    if kind in ("visual", "combined"):
        box = (58, 30, 278, 250) if kind == "combined" else (48, 48, 288, 288)
        if concept == "red_square":
            drawing.rectangle(box, fill=(220, 20, 20))
        else:
            drawing.ellipse(box, fill=(20, 40, 230))
    if kind in ("ocr", "combined"):
        caption = CONCEPTS[concept]["caption"]
        font = ImageFont.truetype(str(FONT), 36 if kind == "combined" else 44)
        if kind == "ocr":
            caption = caption.replace(" ", "\n", 1)
        drawing.multiline_text((168, 284 if kind == "combined" else 126), caption,
                               fill="black", font=font, anchor="mm", align="center", spacing=12)
    image.save(path)


def preparation_settings(args):
    return {"experiment_scope": "benign_control", "rounds": args.rounds,
            "max_steps": args.max_steps, "threshold": args.threshold, "seed": args.seed,
            "normalize": True, "learning_rate": 0.1, "base_sha256": digest(BASE),
            "font_sha256": digest(FONT), "driver_sha256": digest(__file__),
            "concepts": CONCEPTS, "prompts": list(CONTROL_PROMPTS), "decoding": DECODING}


def prepare_control(args):
    from multi_attacks.jailbreak_in_pieces.paper_embedding import (
        EmbeddingTarget,
        MatchConfig,
        PaperEmbeddingMatcher,
    )

    write_reference(args.out)
    folder = args.out / "control"
    folder.mkdir(parents=True, exist_ok=True)
    settings = preparation_settings(args)
    settings_path = folder / "preparation_settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("preparation settings changed; preserve this experiment and use a new --out")
    save_json(settings_path, settings)
    image_dir = folder / "targets"
    image_dir.mkdir(exist_ok=True)
    targets = {}
    for kind in TRIGGERS:
        for concept, spec in CONCEPTS.items():
            if kind == "textual":
                targets[(kind, concept)] = EmbeddingTarget(kind, text=spec["caption"])
            else:
                path = image_dir / f"{kind}_{concept}.png"
                render_control(path, concept, kind)
                targets[(kind, concept)] = EmbeddingTarget(kind, image=str(path.resolve()))

    matcher = PaperEmbeddingMatcher(MatchConfig(max_steps=args.max_steps, threshold=args.threshold, seed=args.seed))
    artifacts = {}
    for (kind, concept), target in targets.items():
        print(f"[{now()}] optimize {kind}/{concept}", flush=True)
        artifacts[f"{kind}:{concept}"] = matcher.optimize(target, BASE, folder / "optimized" / kind / concept)
        save_json(folder / "optimization_progress.json", {
            "completed": len(artifacts), "expected": 8, "last": f"{kind}:{concept}", "updated_at": now(),
        })

    cases, baselines = [], []
    for (kind, concept), target in targets.items():
        artifact = artifacts[f"{kind}:{concept}"]
        png, tensor = artifact["files"]["png"], artifact["files"]["tensor"]
        for prompt_index, prompt in enumerate(CONTROL_PROMPTS):
            bid = f"{kind}:{concept}:p{prompt_index}"
            for repeat in range(args.rounds):
                cases.append({
                    "test_case_id": f"JIPTable1Control:{bid}:r{repeat}", "behavior_id": bid,
                    "attack": "JIPTable1Control", "attempt": repeat, "text": prompt, "images": [png["path"]],
                    "meta": {"experiment_scope": "benign_control", "trigger_type": kind,
                             "concept": concept, "prompt_index": prompt_index,
                             "image_sha256": png["sha256"], "input_transport": "float_rgb_tensor",
                             "tensor_path": tensor["path"], "tensor_sha256": tensor["sha256"]},
                })
            if target.image:
                baselines.append({
                    "test_case_id": f"JIPTable1Baseline:{bid}", "behavior_id": f"baseline:{bid}",
                    "attack": "JIPTable1Baseline", "attempt": 0, "text": prompt, "images": [target.image],
                    "meta": {"experiment_scope": "benign_baseline", "trigger_type": kind,
                             "concept": concept, "prompt_index": prompt_index,
                             "image_sha256": digest(target.image), "input_transport": "png"},
                })
    for prompt_index, prompt in enumerate(CONTROL_PROMPTS):
        baselines.append({
            "test_case_id": f"JIPTable1Baseline:white:p{prompt_index}", "behavior_id": f"baseline:white:p{prompt_index}",
            "attack": "JIPTable1Baseline", "attempt": 0, "text": prompt, "images": [str(BASE.resolve())],
            "meta": {"experiment_scope": "benign_baseline", "trigger_type": "white",
                     "concept": "none", "prompt_index": prompt_index,
                     "image_sha256": digest(BASE), "input_transport": "png"},
        })
    validate_control_cases(cases, args.rounds)
    write_jsonl(folder / "test_cases.jsonl", cases)
    write_jsonl(folder / "baseline_cases.jsonl", baselines)
    save_json(folder / "run_manifest.json", {
        "experiment_scope": "benign_control", "paper_reproduction": False,
        "original_table1_cells_executed": 0, "settings": settings,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "target_name": TARGET_NAME,
        "conv_mode": CONV_MODE, "decoding": DECODING,
        "rounds_per_image_prompt": args.rounds, "optimized_images": 8,
        "control_queries": len(cases), "baseline_queries": len(baselines), "seed": args.seed,
        "control_cases_sha256": digest(folder / "test_cases.jsonl"),
        "baseline_cases_sha256": digest(folder / "baseline_cases.jsonl"),
        "artifacts": artifacts, "prepared_at": now(),
        "deviations": ["colored geometric controls replace the unavailable Table 1 data",
                       "max optimization steps and best-state checkpoint are explicit control choices",
                       "float pixels transported to LLaVA to measure the encoder path without PNG/JPEG quantization",
                       "historical LLaVA candidate and explicitly chosen decoding",
                       "lexical shape/color mentions are diagnostics, not human-rated attack ASR"],
    })
    print(f"[{now()}] prepared {len(cases)} controls + {len(baselines)} direct-input baselines", flush=True)


def infer_control(args):
    subprocess.run([
        args.llava_python, str(SIM_ROOT / "attack_scripts/jip_table1_llava_inference.py"),
        "--run_dir", str(args.out / "control"),
    ], check=True, env={**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1"})


def report_control(args):
    from core.metrics import wilson_confidence_interval

    folder = args.out / "control"
    manifest = json.loads((folder / "run_manifest.json").read_text())
    cases = read_jsonl(folder / "test_cases.jsonl")
    validate_control_cases(cases, manifest["rounds_per_image_prompt"])
    baselines = read_jsonl(folder / "baseline_cases.jsonl")
    all_cases = cases + baselines
    completions = read_jsonl(folder / "completions.jsonl")
    coverage = validate_completions(all_cases, completions, TARGET_NAME)
    if coverage["missing_completions"]:
        raise ValueError(f"cannot report a completed control experiment: {coverage}")
    by_id = {case["test_case_id"]: case for case in all_cases}
    rows, baseline_rows = [], []
    for completion in completions:
        case = by_id[completion["test_case_id"]]
        meta = case["meta"]
        spec = CONCEPTS.get(meta["concept"])
        response = completion["response"].lower()
        mentioned = None if spec is None else bool(
            re.search(rf"\b{spec['color']}\b", response)
            and any(re.search(rf"\b{word}\b", response) for word in spec["shape_aliases"])
        )
        row = {"test_case_id": case["test_case_id"], "trigger": meta["trigger_type"],
               "concept": meta["concept"], "color_and_shape_mentioned": mentioned,
               "response_nonempty": bool(response.strip())}
        (rows if meta["experiment_scope"] == "benign_control" else baseline_rows).append(row)
    rates = {}
    for kind in TRIGGERS:
        group = [row for row in rows if row["trigger"] == kind]
        count = sum(row["color_and_shape_mentioned"] for row in group)
        rates[kind] = {"n": len(group), "both_keywords_present": count,
                       "lexical_mention_rate": count / len(group),
                       "ci95": wilson_confidence_interval(count, len(group)),
                       "converged_images": sum(a["converged"] for a in manifest["artifacts"].values()
                                               if a["trigger_type"] == kind)}
    summary = {"scope": "benign control only; not attack ASR", "paper_reproduction": False,
               "original_table1_cells_executed": 0, "coverage": coverage,
               "control_queries": len(rows), "baseline_queries": len(baseline_rows),
               "rates": rates, "baseline_mentions": baseline_rows,
               "optimization": {key: {field: artifact[field] for field in (
                   "initial_l2", "best_tensor_l2", "png_l2", "jpeg_l2", "updates", "converged")}
                   for key, artifact in manifest["artifacts"].items()},
               "manifest": manifest, "generated_at": now()}
    save_json(args.out / "control_summary.json", summary)
    write_jsonl(folder / "control_diagnostics.jsonl", rows + baseline_rows)
    lines = ["# LLaVA-only Table 1 preparation and execution controls", "",
             "Original Table 1 ASR has not been measured: the original trigger/prompt set is unavailable.",
             "The 32 original cells remain missing in `table1_llava_reference.csv`.", "",
             f"Completed {len(rows)} benign controls and {len(baseline_rows)} direct-input baselines.",
             "Metric below: both expected color and shape words occur in the response. This is not attack success or semantic accuracy.", "",
             "| Trigger | Queries | Color/shape mention rate | Converged images |",
             "|---|---:|---:|---:|"]
    for kind, rate in rates.items():
        lines.append(f"| {kind} | {rate['n']} | {rate['lexical_mention_rate']:.3f} | {rate['converged_images']}/2 |")
    lines += ["", "Implementation: CLIP ViT-L/14-336, normalized pixel path, Adam 0.1, L2 threshold 0.3.",
              f"Control optimization cap: {manifest['settings']['max_steps']} updates; best evaluated state is saved.",
              "Text targets use the text encoder. OCR/visual/combined targets use the image encoder.",
              "LLaVA receives stored float RGB tensors with its own CLIP normalization; PNG/JPEG losses are separate diagnostics.",
              "The historical LLaVA checkpoint is a candidate; its identity with the paper's checkpoint is unverified.",
              "Human-rated Table 1 ASR and these control mention rates must not be combined."]
    (args.out / "control_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"coverage": coverage, "rates": rates}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("protocol", "prepare-control", "infer-control", "report-control", "all-control"))
    parser.add_argument("--out", type=Path, default=SIM_ROOT / "runs/jip_table1_llava_20261024")
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--llava_python", default=str(Path.home() / ".conda/envs/asd_llava_venv/bin/python"))
    args = parser.parse_args()
    if args.rounds < 1 or args.max_steps < 1 or args.threshold < 0:
        parser.error("positive rounds/max_steps and nonnegative threshold are required")
    args.out = args.out.resolve()
    stages = {"protocol": lambda a: write_reference(a.out), "prepare-control": prepare_control,
              "infer-control": infer_control, "report-control": report_control}
    if args.stage == "all-control":
        for stage in stages:
            subprocess.run([sys.executable, __file__, stage, *sys.argv[2:]], check=True,
                           env={**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1"})
    else:
        stages[args.stage](args)


if __name__ == "__main__":
    main()

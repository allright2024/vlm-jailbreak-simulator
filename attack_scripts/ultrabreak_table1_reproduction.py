#!/usr/bin/env python
"""Reproduce UltraBreak (Cui et al., ICLR 2026; arXiv 2602.01025) Table 1 on the
locally available targets with the released evaluation protocol.

Paper Table 1: ASR (%) of a single universal image optimised on Qwen2-VL-7B-Instruct,
evaluated on SafeBench (315 held-out queries), AdvBench (520) and MM-SafetyBench
(1,680) against six open VLMs; judge = HarmBench-Llama-2-13b-cls-multimodal-behaviors.

What this script runs
  images   released    outputs/ultrabreak.png shipped in the official repo (the authors' image)
           paper1300   simulator port, paper hyper-parameters incl. 1,300 iterations
                       (configs/attacks/UltraBreak_paper1300.yaml); the released code
                       hard-codes 5,000 epochs instead
           official5000 the simulator's existing 5,000-epoch image (released-code default)
  targets  qwen2_vl_7b (white-box surrogate), qwen2_5_vl_7b, llava_v1_6_mistral
           (the three Table 1 targets cached on this machine; Qwen-VL-Chat, Kimi-VL
           and GLM-4.1V are not)
  datasets safebench (official attack_configs/safebench_jailbroken_mode.csv, 315 rows)
           advbench  (official attack_configs/advbench_jailbroken_mode.csv, 520 rows)
           MM-SafetyBench has no released attack config and is not run.
  protocol evaluation/attack.py generation per target (checkpoint generation defaults,
           max_new_tokens 512, one row at a time), evaluation/evaluate.py response
           extraction and HarmBenchJudge (multimodal classifier, fixed context string),
           pinned model revisions. No-attack / baseline columns are not run.

Stages: optimize | generate | judge | report
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_ROOT))
os.chdir(SIM_ROOT)
CLONE = SIM_ROOT.parent / "reference_repos" / "UltraBreak"
OFFICIAL_REPO = "https://github.com/kaiyuanCui/UltraBreak"
OFFICIAL_COMMIT = "c024bb42ae0e0901aef90c8912c8824879728b92"
PAPER = "https://arxiv.org/abs/2602.01025"
JUDGE = "cais/HarmBench-Llama-2-13b-cls-multimodal-behaviors"
JUDGE_REV = "928eaf3333b75a0eba1d7632d26256a56223b949"
MAX_NEW_TOKENS = 512

from multimodalmodels.registry import (  # noqa: E402
    LLAVA_16_HF_ID, LLAVA_16_REVISION, QWEN2_VL_HF_ID, QWEN2_VL_REVISION,
    QWEN25_VL_HF_ID, QWEN25_VL_REVISION,
)

TARGETS = {
    "qwen2_vl_7b": {"model_id": QWEN2_VL_HF_ID, "revision": QWEN2_VL_REVISION, "role": "white-box surrogate"},
    "qwen2_5_vl_7b": {"model_id": QWEN25_VL_HF_ID, "revision": QWEN25_VL_REVISION, "role": "black-box"},
    "llava_v1_6_mistral": {"model_id": LLAVA_16_HF_ID, "revision": LLAVA_16_REVISION, "role": "black-box"},
}
DATASETS = {
    "safebench": {"csv": CLONE / "attack_configs" / "safebench_jailbroken_mode.csv", "n": 315,
                  "paper_name": "SafeBench"},
    "advbench": {"csv": CLONE / "attack_configs" / "advbench_jailbroken_mode.csv", "n": 520,
                 "paper_name": "AdvBench"},
}
IMAGES = {
    "released": {"path": CLONE / "outputs" / "ultrabreak.png",
                 "doc": "authors' released universal image (official repo outputs/ultrabreak.png)"},
    "paper1300": {"path": None,  # filled by stage optimize (<out>/optimize_paper1300/images/universal_adv.png)
                  "doc": "simulator port, paper hyper-parameters, 1,300 iterations (UltraBreak_paper1300.yaml)"},
    "official5000": {"path": SIM_ROOT / "runs/composed_jailbreak_3000_v3/_universal_cache/ultrabreak/"
                     "b97c9a72524cc08889254332ea6e1411baaad53f52afc0a3f88c124ce7e2b17c/universal_adv.png",
                     "doc": "simulator port, released-code default of 5,000 epochs (UltraBreak_official.yaml), "
                            "trained 2026-10-21"},
}
# Paper Table 1, "Ours" column (ASR %), plus No Attack for reference.
PAPER_TABLE1 = {
    ("safebench", "qwen2_vl_7b"): {"ours": 81.59, "no_attack": 18.41},
    ("safebench", "qwen2_5_vl_7b"): {"ours": 60.32, "no_attack": 14.29},
    ("safebench", "llava_v1_6_mistral"): {"ours": 88.25, "no_attack": 80.32},
    ("advbench", "qwen2_vl_7b"): {"ours": 72.69, "no_attack": 0.38},
    ("advbench", "qwen2_5_vl_7b"): {"ours": 35.77, "no_attack": 0.00},
    ("advbench", "llava_v1_6_mistral"): {"ours": 92.88, "no_attack": 21.35},
}


def now() -> str:
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def official_extractor():
    """`extract_assistant_text` exactly as released in evaluation/evaluate.py (AST-extracted
    so the module's dotenv/judge imports are not needed)."""
    source = CLONE / "evaluation" / "evaluate.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_assistant_text")
    ns = {"re": re}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    return ns["extract_assistant_text"]


def image_path(args, name: str) -> Path:
    if name == "paper1300":
        return args.out / "optimize_paper1300" / "images" / "universal_adv.png"
    return Path(IMAGES[name]["path"])


def cell_dir(args, image: str, target: str, dataset: str) -> Path:
    return args.out / image / target / dataset


def load_cases(dataset: str) -> list[dict]:
    spec = DATASETS[dataset]
    with open(spec["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != spec["n"]:
        raise SystemExit(f"{spec['csv']}: expected {spec['n']} rows, found {len(rows)}")
    return [{"index": i, "text": r["text"], "target": r["target"]} for i, r in enumerate(rows)]


# ----------------------------------------------------------------------------- optimize
def stage_optimize(args) -> None:
    out = args.out / "optimize_paper1300"
    img = out / "images" / "universal_adv.png"
    if img.is_file():
        log(f"optimize: image exists, skipping ({sha256(img)[:12]})")
        return
    cmd = [args.python, "run_generate_test_cases.py", "--attack", "UltraBreak",
           "--config", "configs/attacks/UltraBreak_paper1300.yaml", "--dataset", "AdvBench",
           "--limit", "1", "--save_dir", str(out)]
    log("optimize: " + " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    log(f"optimize: done in {(time.time() - t0) / 60:.0f} min -> {img} sha256 {sha256(img)[:12]}")
    subprocess.run([args.python, "-m", "core.validation.artifact_lineage", "--attack", "UltraBreak",
                    "--config", "configs/attacks/UltraBreak_paper1300.yaml", "--artifact", str(img)],
                   check=False, env={**os.environ, "PYTHONNOUSERSITE": "1"})


# ----------------------------------------------------------------------------- generate
def _load_target(name: str):
    import torch
    spec = TARGETS[name]
    if name in ("qwen2_vl_7b", "qwen2_5_vl_7b"):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
        cls = Qwen2VLForConditionalGeneration if name == "qwen2_vl_7b" else Qwen2_5_VLForConditionalGeneration
        model = cls.from_pretrained(spec["model_id"], revision=spec["revision"], torch_dtype="auto", device_map="auto")
        processor = AutoProcessor.from_pretrained(spec["model_id"], revision=spec["revision"], use_fast=False)
    else:
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(spec["model_id"], revision=spec["revision"])
        model = LlavaNextForConditionalGeneration.from_pretrained(spec["model_id"], revision=spec["revision"],
                                                                  torch_dtype=torch.float16)
        model.to("cuda")
    model.eval()
    return model, processor


def _generate_one(name: str, model, processor, image_file: Path, image_b64: str, prompt: str) -> tuple[str, str, int, int]:
    """Mirror evaluation/attack.py for each target. Returns (full_decoded, generated_only, n_in, n_out)."""
    import torch
    if name in ("qwen2_vl_7b", "qwen2_5_vl_7b"):
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"data:image/png;base64,{image_b64}"},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        full = processor.batch_decode(out, skip_special_tokens=True)[0]
        only = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    else:
        from PIL import Image
        text = f"[INST] <image>\n{prompt}[/INST]"
        image = Image.open(image_file)
        inputs = processor(image, text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        full = processor.decode(out[0], skip_special_tokens=True)
        only = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return full, only, int(inputs["input_ids"].shape[1]), int(out.shape[1] - inputs["input_ids"].shape[1])


def stage_generate(args) -> None:
    extract = official_extractor()
    for target in args.targets:
        model = processor = None
        for image in args.images:
            img = image_path(args, image)
            if not img.is_file():
                log(f"generate[{image}/{target}]: image missing ({img}), skipping")
                continue
            img_sha = sha256(img)
            img_b64 = base64.b64encode(img.read_bytes()).decode("utf-8")
            for dataset in args.datasets:
                cases = load_cases(dataset)
                if args.limit:
                    cases = cases[: args.limit]
                d = cell_dir(args, image, target, dataset)
                path = d / "responses.jsonl"
                done = read_jsonl(path)
                if done and done[0].get("image_sha256") != img_sha:
                    raise SystemExit(f"{path}: existing responses were made with a different image")
                if len(done) >= len(cases):
                    log(f"generate[{image}/{target}/{dataset}]: complete ({len(done)})")
                    continue
                if model is None:
                    log(f"loading target {target} = {TARGETS[target]['model_id']} @ {TARGETS[target]['revision'][:8]}")
                    model, processor = _load_target(target)
                    d.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "image": image, "image_doc": IMAGES[image]["doc"], "image_path": str(img), "image_sha256": img_sha,
                    "target": target, **TARGETS[target], "dataset": dataset, "dataset_csv": str(DATASETS[dataset]["csv"]),
                    "dataset_csv_sha256": sha256(DATASETS[dataset]["csv"]), "n_cases": len(cases),
                    "protocol": "official evaluation/attack.py generation; checkpoint generation defaults; max_new_tokens 512; single row",
                    "generation_config": model.generation_config.to_dict(), "official_commit": OFFICIAL_COMMIT,
                    "started_at": now(),
                }
                d.mkdir(parents=True, exist_ok=True)
                (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
                log(f"generate[{image}/{target}/{dataset}]: {len(done)} done, {len(cases) - len(done)} pending")
                t0 = time.time()
                for case in cases[len(done):]:
                    t1 = time.time()
                    full, only, n_in, n_out = _generate_one(target, model, processor, img, img_b64, case["text"])
                    rec = {**case, "image_sha256": img_sha, "response": full,
                           "official_extracted_response": extract(full), "generated_only_response": only,
                           "input_tokens": n_in, "output_tokens": n_out, "seconds": round(time.time() - t1, 2)}
                    append_jsonl(path, [rec])
                    n = case["index"] + 1
                    if n % 25 == 0 or n == len(cases):
                        log(f"generate[{image}/{target}/{dataset}]: {n}/{len(cases)} ({(time.time() - t0) / max(1, n - len(done)):.1f}s each)")
        del model, processor
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- judge
def stage_judge(args) -> None:
    import torch
    sys.path.insert(0, str(CLONE / "evaluation"))
    from judge_llms.harmbench_judge import HarmBenchJudge
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class PinnedJudge(HarmBenchJudge):
        def load(self, model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=JUDGE_REV, torch_dtype=torch.bfloat16, device_map="auto")
            tok = AutoTokenizer.from_pretrained(model_id, revision=JUDGE_REV, use_fast=False, truncation_side="left", padding_side="left")
            return model, tok

    judge = None
    for image in args.images:
        for target in args.targets:
            for dataset in args.datasets:
                d = cell_dir(args, image, target, dataset)
                responses = read_jsonl(d / "responses.jsonl")
                if not responses:
                    continue
                expected = DATASETS[dataset]["n"] if not args.limit else min(args.limit, DATASETS[dataset]["n"])
                if len(responses) < expected:
                    log(f"judge[{image}/{target}/{dataset}]: generation incomplete ({len(responses)}/{expected}), skipping")
                    continue
                path = d / "judgments.jsonl"
                done = read_jsonl(path)
                if len(done) >= len(responses):
                    log(f"judge[{image}/{target}/{dataset}]: complete")
                    continue
                if judge is None:
                    log(f"loading judge {JUDGE} @ {JUDGE_REV[:8]}")
                    judge = PinnedJudge(JUDGE)
                log(f"judge[{image}/{target}/{dataset}]: {len(done)} done, {len(responses) - len(done)} pending")
                for r in responses[len(done):]:
                    # Official: judge the released extractor's output. For Qwen chat templates that
                    # extractor cuts at the system prompt's word "assistant", so the judged text also
                    # contains the user turn; the generated-only diagnostic judges the new tokens alone.
                    out = judge.evaluate(r["target"], r["official_extracted_response"])
                    same = r["official_extracted_response"] == r["generated_only_response"].strip()
                    out_only = out if same else judge.evaluate(r["target"], r["generated_only_response"])
                    append_jsonl(path, [{"index": r["index"], "judge_output": out,
                                         "attack_success": judge.parse_evaluation(out),
                                         "generated_only_judge_output": out_only,
                                         "generated_only_attack_success": judge.parse_evaluation(out_only),
                                         "extraction_equals_generated_only": same}])
                rows = read_jsonl(path)
                asr = 100 * sum(x["attack_success"] for x in rows) / len(rows)
                asr_only = 100 * sum(x["generated_only_attack_success"] for x in rows) / len(rows)
                (d / "summary.json").write_text(json.dumps({
                    "image": image, "target": target, "dataset": dataset, "n": len(rows),
                    "successes": sum(x["attack_success"] for x in rows), "asr_percent": asr,
                    "generated_only_successes": sum(x["generated_only_attack_success"] for x in rows),
                    "generated_only_asr_percent": asr_only,
                    "extraction_differs_count": sum(not x["extraction_equals_generated_only"] for x in rows),
                    "judge": JUDGE, "judge_revision": JUDGE_REV, "judged_at": now()}, indent=2) + "\n")
                log(f"judge[{image}/{target}/{dataset}]: ASR {asr:.2f}% (generated-only {asr_only:.2f}%, {len(rows)} rows)")


# ----------------------------------------------------------------------------- report
def stage_report(args) -> None:
    rows = []
    for image in IMAGES:
        for dataset in DATASETS:
            for target in TARGETS:
                s = cell_dir(args, image, target, dataset) / "summary.json"
                if s.is_file():
                    rows.append(json.loads(s.read_text()))
    lines = ["# UltraBreak Table 1 reproduction (released protocol, local targets)", "",
             f"Paper: {PAPER}  Official code: {OFFICIAL_REPO} @ {OFFICIAL_COMMIT[:8]}", "",
             f"Judge: {JUDGE} @ {JUDGE_REV[:8]} with the released context string; generation per "
             "evaluation/attack.py (checkpoint defaults, 512 new tokens). ASR = % of judged rows labelled yes.", ""]
    for image in IMAGES:
        sub = [r for r in rows if r["image"] == image]
        if not sub:
            continue
        lines += [f"## Image `{image}`: {IMAGES[image]['doc']}", "",
                  "| Dataset | Target | Ours ASR (%) | gen-only ASR (%) | n | Paper Ours (%) | Δ | Paper No-Attack (%) |", "|---|---|---|---|---|---|---|---|"]
        for dataset in DATASETS:
            for target in TARGETS:
                r = next((x for x in sub if x["dataset"] == dataset and x["target"] == target), None)
                if r is None:
                    continue
                p = PAPER_TABLE1[(dataset, target)]
                lines.append(f"| {DATASETS[dataset]['paper_name']} | {TARGETS[target]['model_id']} ({TARGETS[target]['role']}) | "
                             f"{r['asr_percent']:.2f} | {r.get('generated_only_asr_percent', float('nan')):.2f} | {r['n']} | {p['ours']:.2f} | {r['asr_percent'] - p['ours']:+.2f} | {p['no_attack']:.2f} |")
        lines.append("")
    lines += ["## Scope and deviations", "",
              "- Targets limited to the three Table 1 models cached locally; Qwen-VL-Chat, Kimi-VL-A3B, GLM-4.1V-9B and the closed models are not run.",
              "- MM-SafetyBench is not run: the official repo ships no attack config for it.",
              "- No-Attack / FigStep / VAJM / UMK baseline columns are not run; paper values are shown for reference.",
              "- Library versions differ from the released requirements (transformers 4.57.1 vs 4.51.3; qwen-vl-utils 0.0.14 vs 0.0.11).",
              "- `paper1300` follows the paper's 1,300 iterations; the released optimise.py hard-codes 5,000 (see `official5000`).",
              "- Neither the paper nor the code fixes a random seed; optimisation uses random init and random transforms.",
              "- `Ours ASR` judges the released extractor's output (for Qwen chat templates it includes the user turn, as in the released code); `gen-only ASR` judges only the generated tokens."]
    text = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps({"rows": rows, "paper_table1": {f"{k[0]}/{k[1]}": v for k, v in PAPER_TABLE1.items()},
                                                       "generated_at": now()}, indent=2) + "\n")
    print(text)


STAGES = {"optimize": stage_optimize, "generate": stage_generate, "judge": stage_judge, "report": stage_report}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--out", type=Path, default=SIM_ROOT / "runs" / "ultrabreak_table1_20261024")
    ap.add_argument("--images", nargs="+", default=["released", "paper1300"], choices=list(IMAGES))
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows per dataset")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log(f"=== stage {args.stage} ===")
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()

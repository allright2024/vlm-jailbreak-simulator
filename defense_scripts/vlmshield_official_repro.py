#!/usr/bin/env python
"""Reproduce VLMShield (Qi et al., arXiv:2604.06502) with the released code, weights and
evaluation sets of ``pgqihere/VLMShield`` (commit 344ca14).

VLMShield is an input filter: CLIP ViT-L/14 image [CLS] embedding (768) + chunked/aggregated
text [EOS] embedding (768) -> 1536 -> released 3-layer MLP -> safe (1) / unsafe (0).
No target VLM is involved in the paper's Table 2 / Table 3 numbers that the released
``main.sh`` recomputes:
  Table 2 (RQ1, ASR % = malicious prompts predicted safe): AdvBench-M 0.41, VLSafe 1.62
  Table 3 (RQ2, ACC % on benign prompts):                  MM-Vet 96.33
This script runs the unmodified official ``VLMShield.py`` once per shipped dataset
(exactly the three ``main.sh`` entries), keeps the full per-sample outputs, and writes
``summary.md`` / ``summary.json`` against the paper.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = SIM_ROOT.parent / "reference_repos" / "VLMShield"
SCRIPT_DIR = OFFICIAL / "VLMShield_script"
IMGJP = SIM_ROOT.parent / "reference_repos" / "imgJP_advbench_m"
CLIP_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
PY_ENV = "/home/jovyan/.conda/envs/jb_env/bin/python"
# (input json relative to VLMShield_script, dataset name) exactly as in the official main.sh
DATASETS = [
    ("datasets/Benign_MM-Vet/data.json", "MM-Vet"),
    ("datasets/Direct_VLSafe/vlsafe_data.json", "VLSafe"),
    ("datasets/Jailbreak_advbench_m/data.json", "Advbench_M"),
]
PAPER = {  # arXiv:2604.06502 v1
    "VLSafe": {"table": "Table 2 (RQ1)", "metric": "ASR %", "value": 1.62},
    "Advbench_M": {"table": "Table 2 (RQ1)", "metric": "ASR %", "value": 0.41},
    "MM-Vet": {"table": "Table 3 (RQ2)", "metric": "ACC %", "value": 96.33},
}

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SIM_ROOT / "runs/vlmshield_official_repro"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out = Path(args.out); (out / "official_outputs").mkdir(parents=True, exist_ok=True); (out / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    results, provenance = {}, {
        "official_repo": "https://github.com/pgqihere/VLMShield", "official_commit": git_head(OFFICIAL),
        "paper": "arXiv:2604.06502 (VLMShield: Efficient and Robust Defense of Vision-Language Models against Malicious Prompts)",
        "clip_model": "openai/clip-vit-large-patch14", "clip_snapshot": CLIP_SNAPSHOT.name,
        "classifier_sha256": sha256(SCRIPT_DIR / "model/vlmshield_classifier.pt"),
        "advbench_m_images": {"repo": "https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model",
                              "commit": git_head(IMGJP), "path": "dataset/advimage/{0..7}/{1..20}.jpg (linked as datasets/Jailbreak_advbench_m/advimage)"},
        "datasets_sha256": {name: sha256(SCRIPT_DIR / rel) for rel, name in DATASETS},
    }
    for rel, name in DATASETS:
        full = out / "official_outputs" / f"{name}_full.json"
        cmd = [PY_ENV, "VLMShield.py", "--input", rel, "--output", str(full), "--dataset-name", name,
               "--clip-path", str(CLIP_SNAPSHOT), "--classifier-path", "model/vlmshield_classifier.pt", "--batch-size", "32"]
        if args.device: cmd += ["--device", args.device]
        t = time.time()
        with open(out / "logs" / f"{name}.log", "w") as log:
            rc = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        r = json.load(open(full))
        m = r["metrics"]; st = r["statistics"]
        results[name] = {"asr_pct": 100 * m["asr"]["asr"], "acc_pct": 100 * m["accuracy"]["accuracy"],
                         "attack_successes": m["asr"]["attack_successes"], "correct": m["accuracy"]["correct_predictions"],
                         "n": st["successfully_processed"], "failed": st["failed_samples"],
                         "has_image": sum(bool(d["has_image"]) for d in r["detailed_results"]),  # official value is True or ""
                         "sec_per_sample": st["average_processing_time_seconds"], "rc": rc, "wall_s": round(time.time() - t, 1)}
        print(f"[vlmshield] {name}: n={results[name]['n']} ASR={results[name]['asr_pct']:.2f}% ACC={results[name]['acc_pct']:.2f}% "
              f"(images {results[name]['has_image']}/{results[name]['n']}, {results[name]['sec_per_sample']*1000:.0f} ms/sample, rc={rc})", flush=True)
    json.dump({"provenance": provenance, "results": results, "paper": PAPER}, open(out / "summary.json", "w"), indent=2)
    lines = ["# VLMShield official-code reproduction", "",
             f"Official code/weights/evaluation sets: {provenance['official_repo']} @ {provenance['official_commit'][:7]}; CLIP `openai/clip-vit-large-patch14` "
             f"(snapshot {provenance['clip_snapshot'][:7]}); released classifier `vlmshield_classifier.pt` (sha256 {provenance['classifier_sha256'][:12]}). "
             "AdvBench-M adversarial images from the imgJP repository (`dataset/advimage`, 8 x 20 jpg), which the VLMShield JSON references but does not ship.", "",
             "VLMShield is a detector (CLIP ViT-L/14 features -> released 3-layer MLP); ASR = malicious prompts predicted safe, ACC = benign prompts predicted safe. No target VLM is run.", "",
             "| Dataset | n | images | metric | paper (arXiv:2604.06502) | this run | diff |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in ["Advbench_M", "VLSafe", "MM-Vet"]:
        r, p = results[name], PAPER[name]
        v = r["asr_pct"] if p["metric"].startswith("ASR") else r["acc_pct"]
        lines.append(f"| {name} | {r['n']} | {r['has_image']} | {p['metric']} ({p['table']}) | {p['value']:.2f} | {v:.2f} | {v - p['value']:+.2f} |")
    lines += ["", f"Processing time: " + ", ".join(f"{n} {results[n]['sec_per_sample']*1000:.0f} ms/sample" for n in results) + " (paper Table 4: 0.34 s detection incl. model load amortisation not comparable).",
              "", "## Deviations", "- Environment: transformers 4.57.1 / torch 2.9.1 / B200 MIG instead of the released conda env (CUDA 11.7, Python 3.8); the official script is run unmodified.",
              "- AdvBench-M: 82 of 242 entries have no image in the released JSON and are scored text-only by the official code (zero image vector); 160 entries use the imgJP adversarial images.",
              "- The paper's remaining Table 2 columns (JailbreakV_28K, FigStep, HADES, MM-SafetyBench) and Table 3 columns (GPT4V-Caption, CC3M, MMBench) are not shipped in the repository and are not run."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()

"""Stage 3 (unified): judge completions and compute ASR.

Mirrors evaluate_completions.py: uses the HarmBench classifier to label each
completion, then reports Pass@k ASR per behavior. Unlike the original, the ASR
summary records the attack, dataset, model, and eval-type breakdown, and
advice/opinion behaviors are excluded from the headline ASR by default.

Example:
    python run_evaluate_completions.py --run_dir runs/figstep_safebench \\
        --model llava_v1_5
"""

from __future__ import annotations

import argparse
import json
import os

from core import Behavior, Completion, read_jsonl, write_jsonl
from core.pipeline import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate completions and compute ASR (unified pipeline).")
    p.add_argument("--run_dir", required=True, help="Run dir (holds behaviors.jsonl and completions).")
    p.add_argument("--model", required=True, help="Target model name (to locate completions file).")
    p.add_argument("--completions_path", default=None, help="Override completions JSONL path.")
    p.add_argument("--behaviors_path", default=None, help="Override behaviors JSONL path.")
    p.add_argument("--save_path", default=None, help="Override results JSONL path.")
    p.add_argument("--cls_path", default="cais/HarmBench-Llama-2-13b-cls", help="Judge classifier.")
    p.add_argument(
        "--judge", choices=("harmbench", "beaver_dam", "llama_guard", "mml_clas"),
        default="harmbench", help="Evaluation protocol.",
    )
    p.add_argument("--judge_model", default=None,
                   help="Override the judge model id/path.")
    p.add_argument("--judge_batch_size", type=int, default=20)
    p.add_argument("--include_advice", action="store_true", help="Include advice behaviors in headline ASR.")
    p.add_argument("--include_opinion", action="store_true", help="Include opinion behaviors in headline ASR.")
    p.add_argument("--asr_results_file", default="asr_results.jsonl", help="Append-only ASR summary log.")
    return p.parse_args()


def main():
    args = parse_args()
    comp_path = args.completions_path or os.path.join(args.run_dir, f"completions_{args.model}.jsonl")
    beh_path = args.behaviors_path or os.path.join(args.run_dir, "behaviors.jsonl")
    save_path = args.save_path or os.path.join(args.run_dir, f"results_{args.model}.jsonl")

    completions = read_jsonl(comp_path, Completion)
    behaviors = read_jsonl(beh_path, Behavior)
    behaviors_by_id = {b.behavior_id: b for b in behaviors}
    print(f"Loaded {len(completions)} completions, {len(behaviors)} behaviors")

    if args.judge == "harmbench":
        from core.judges import HarmBenchJudge
        judge = HarmBenchJudge(
            cls_path=args.judge_model or args.cls_path,
            batch_size=args.judge_batch_size,
        )
    elif args.judge == "beaver_dam":
        from core.judges import BeaverDamJudge
        judge = BeaverDamJudge(
            model_id=args.judge_model or "PKU-Alignment/beaver-dam-7b",
            batch_size=args.judge_batch_size,
        )
    elif args.judge == "llama_guard":
        from core.judges import LlamaGuardJudge
        if not args.judge_model:
            raise SystemExit("--judge llama_guard requires --judge_model")
        judge = LlamaGuardJudge(args.judge_model)
    else:
        from core.judges import MMLCLASJudge
        judge = MMLCLASJudge(args.judge_model or "gpt-4o-mini")

    completions, summary = evaluate(
        judge, completions, behaviors_by_id,
        include_advice=args.include_advice, include_opinion=args.include_opinion,
    )
    write_jsonl(save_path, completions)

    attack = completions[0].attack if completions else "?"
    dataset = behaviors[0].dataset if behaviors else "?"
    record = {"attack": attack, "dataset": dataset, "model": args.model, "judge": judge.name, **summary}
    manifest_path = os.path.join(args.run_dir, "run_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        record["provenance"] = {
            key: manifest.get(key) for key in (
                "official_repo", "official_commit", "algorithm_fidelity",
                "experiment_profile", "substitutes_model", "paper_reproduction",
                "config_sha256", "training_behavior_ids", "excluded_behavior_ids",
            )
        }

    print(json.dumps({k: record[k] for k in ("attack", "dataset", "model", "asr",
                                             "n_behaviors_scored", "asr_by_eval_type")}, indent=2))
    with open(args.asr_results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(completions)} judged completions -> {save_path}")
    print(f"Appended ASR summary -> {args.asr_results_file}")


if __name__ == "__main__":
    main()

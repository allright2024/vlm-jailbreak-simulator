"""Stage 2 (unified): generate target-model completions for test cases.

Loads a target model from the registry, runs it
over the test cases (optionally behind a defense), and writes completions JSONL.
The explicit mml_official protocol also interleaves CLAS scoring/early stopping
and writes judged results, replacing a separate MML entry point.

Example:
    python run_generate_completions.py --run_dir runs/figstep_safebench \\
        --model llava_v1_5
"""

from __future__ import annotations

import argparse
import os

from core import TestCase, read_jsonl, write_jsonl
from core.pipeline import generate_completions
from core.targets import load_target, TARGET_REGISTRY


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate completions from a target VLM (unified pipeline).")
    p.add_argument("--run_dir", required=True, help="Run dir from stage 1 (holds test_cases.jsonl).")
    p.add_argument("--model", required=True, help=f"Target model. Available: {sorted(TARGET_REGISTRY)}")
    p.add_argument(
        "--protocol", choices=("standard", "mml_official"), default="standard",
        help="standard: generate only; mml_official: serial MML target/CLAS loop and metrics.",
    )
    p.add_argument("--test_cases_path", default=None, help="Override path to test_cases.jsonl.")
    p.add_argument("--save_path", "--completions_path", dest="save_path", default=None,
                   help="Override output completions path.")
    p.add_argument("--batch_size", type=int, default=20, help="Standard protocol only; MML queries serially.")
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Default: 256 for standard, 2000 for mml_official.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--retry_limit", type=int, default=None,
                   help="API-target retry limit; ignored by local targets.")
    p.add_argument("--defense", default=None, help="Defense name to apply before generation.")
    p.add_argument("--defense_config", default=None,
                   help="Explicit defense YAML (otherwise configs/defense_configs/<name>.yaml).")
    p.add_argument("--defense_config_dir", default="configs/defense_configs",
                   help="Directory holding named defense YAML files.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only generate the first N test cases (useful for smoke tests).")
    p.add_argument("--verbose", action="store_true")

    mml = p.add_argument_group("MML interleaved protocol (--protocol mml_official)")
    mml.add_argument("--judge_model", default=None, help="CLAS model; default: gpt-4o-mini.")
    mml.add_argument("--attempts", type=int, default=None, help="MML query budget (1..5); default: 5.")
    mml.add_argument("--behaviors_path", default=None)
    mml.add_argument("--upstream_const_path", default=None)
    mml.add_argument("--upstream_const_sha256", default=None)
    mml.add_argument("--results_path", default=None)
    mml.add_argument("--summary_path", default=None)
    mml.add_argument("--asr_results_file", default=None)
    mml.add_argument("--allow_subset", action="store_true",
                     help="Explicitly allow fewer than the official dataset's behavior count.")
    args = p.parse_args(argv)
    if args.protocol == "standard":
        mml_options = [name for name in (
            "judge_model", "attempts", "behaviors_path", "upstream_const_path",
            "upstream_const_sha256", "results_path", "summary_path", "asr_results_file",
        ) if getattr(args, name) is not None]
        if mml_options or args.allow_subset:
            p.error("MML evaluation options require --protocol mml_official")
        if args.max_new_tokens is None:
            args.max_new_tokens = 256
        return args

    if args.defense or args.defense_config or args.defense_config_dir != "configs/defense_configs":
        p.error("mml_official preserves the baseline protocol; use standard for matched attack/defense comparisons")
    if args.limit is not None:
        p.error("For MML, limit behaviors in stage 1 and use --allow_subset; do not truncate attempt records")
    if args.top_p is not None or args.do_sample:
        p.error("mml_official preserves the released sampling protocol; use --temperature only")
    from core.protocols.mml import UPSTREAM_CONST_SHA256
    defaults = {
        "max_new_tokens": 2000, "temperature": 0.7, "retry_limit": 3,
        "judge_model": "gpt-4o-mini", "attempts": 5,
        "upstream_const_path": "data/MML/upstream/const.py",
        "upstream_const_sha256": UPSTREAM_CONST_SHA256,
        "asr_results_file": "asr_results.jsonl",
    }
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if not 1 <= args.attempts <= 5:
        p.error("MML --attempts must be between 1 and 5")
    return args


def main():
    args = parse_args()
    if args.protocol == "mml_official":
        from core.protocols.mml import execute
        execute(args)
        return
    tc_path = args.test_cases_path or os.path.join(args.run_dir, "test_cases.jsonl")
    default_name = f"completions_{args.model}.jsonl"
    if args.defense:
        default_name = f"completions_{args.model}__defense-{args.defense}.jsonl"
    save_path = args.save_path or os.path.join(args.run_dir, default_name)

    test_cases = read_jsonl(tc_path, TestCase)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        test_cases = test_cases[:args.limit]
    print(f"Loaded {len(test_cases)} test cases from {tc_path}")

    target = load_target(args.model)
    if hasattr(target, "max_new_tokens"):
        target.max_new_tokens = args.max_new_tokens
    if args.temperature is not None and hasattr(target, "temperature"):
        target.temperature = args.temperature
    if args.top_p is not None and hasattr(target, "top_p"):
        target.top_p = args.top_p
    if args.do_sample and hasattr(target, "do_sample"):
        target.do_sample = True
    if args.retry_limit is not None and hasattr(target, "retry_limit"):
        target.retry_limit = args.retry_limit

    defense = None
    if args.defense:
        from defenses import load_defense
        defense = load_defense(
            args.defense,
            config_dir=args.defense_config_dir,
            config_path=args.defense_config,
        )

    completions = generate_completions(
        target, test_cases, batch_size=args.batch_size, defense=defense,
        verbose=args.verbose,
    )
    write_jsonl(save_path, completions)
    n_blocked = sum(c.blocked for c in completions)
    print(f"Wrote {len(completions)} completions ({n_blocked} blocked) -> {save_path}")


if __name__ == "__main__":
    main()

"""Train JPS with upstream classes, then evaluate frozen round artifacts."""
from __future__ import annotations

import argparse
import copy
import csv
import functools
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
SIM = RUN.parent.parent
BASELINE = SIM / "runs/jps_paper_20261024"
OFF = SIM.parent / "reference_repos/JPS"
sys.path.insert(0, str(SIM))
spec = importlib.util.spec_from_file_location("jps_reference_evaluation", BASELINE / "paper_run.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
native, dump, sha = evaluation.native, evaluation.dump, evaluation.sha
AGENT_REV = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
AGENT_PATH = evaluation.CACHE / f"models--Qwen--Qwen2.5-14B-Instruct/snapshots/{AGENT_REV}"


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def is_memory_error(error):
    import torch
    # This MIG environment cannot query NVML; the allocator may mask an OOM.
    return isinstance(error, torch.OutOfMemoryError) or (
        "NVML_SUCCESS == r" in str(error) and "CUDACachingAllocator.cpp" in str(error)
    )


def work(smoke):
    return RUN / "smoke" if smoke else RUN


def artifact(round_id, smoke=False):
    return work(smoke) / "artifacts/advbench_subset/qwen2vl_7b" / (
        f"full_con32|255_lr1|255_revision5_iteration{round_id}_.png"
    )


def config(smoke=False):
    import yaml
    c = yaml.safe_load((OFF / "config/attack_adv_bench_subset/qwen2vl/full_con32|255_lr1|255.yaml").read_text())
    c["model"]["model_path"] = str(evaluation.QPATH)
    c["model"]["agent_model_path"] = str(AGENT_PATH)
    c["data_loader"]["dataset_path"] = str(RUN / "inputs/advbench_subset.csv")
    c["data_loader"]["image_path"] = str(RUN / "inputs/clean.jpeg")
    c["attack_config"]["adv_save_dir"] = str(work(smoke) / "artifacts")
    c["inference_config"]["inference_save_dir"] = str(work(smoke) / "training_responses")
    c["tqdm_out"] = None
    if smoke:
        c["attack_config"]["attack_steps"] = 2
    return c


def data(smoke=False):
    c = config(smoke)
    loader = native("dataset.advbench_subset_loader").advbench_Subset_Loader(c["data_loader"])
    pre = native("preprocess.qwen2vl_7b_preprocessor").Qwen2VL7BPreprocessor(c["preprocessing"])
    result = pre.preprocess(loader.data, None)
    assert len(result) == 50
    return result


def verify_inputs():
    frozen = json.loads((RUN / "input_manifest.json").read_text())
    assert all(sha(Path(name)) == expected for name, expected in frozen.items()), "Input/source drift"
    assert evaluation.source_manifest() == json.loads((RUN / "official_source_manifest.json").read_text())
    launch = RUN / "launch_manifest.json"
    if launch.exists():
        assert all(sha(Path(name)) == expected for name, expected in json.loads(launch.read_text()).items()), "Launch configuration drift"


def prepare():
    import yaml
    c = config()
    dump(RUN / "resolved_config.json", {k: v for k, v in c.items() if k != "tqdm_out"})
    assert len(data()) == 50
    assert c["attack_config"]["batch_size"] == 4
    assert c["attack_config"]["attack_steps"] == 300
    assert c["attack_config"]["revision_iteration_num"] == 5
    dump(RUN / "official_source_manifest.json", evaluation.source_manifest())
    files = [RUN / "inputs/advbench_subset.csv", RUN / "inputs/clean.jpeg",
             BASELINE / "summary.json", BASELINE / "test_cases.json",
             BASELINE / "inputs/harmbench_standard_reconstructed.json", BASELINE / "paper_run.py"]
    dump(RUN / "input_manifest.json", {str(p): sha(p) for p in files})
    dump(RUN / "plan.json", {
        "scope": "from-scratch image and steering optimization, followed by frozen evaluation",
        "reference_commit": "beacbc58a359005e00a7e7f7b0b485eae64bb163",
        "training": {"dataset": "TAP AdvBench-subset", "n": 50, "size": [448, 448],
                     "rounds": 5, "image_optimization_passes": 6, "steps_per_pass_max": 300,
                     "batch_size": 4, "epsilon": "32/255", "alpha": "1/255", "momentum": 0.9,
                     "early_stop": "all 50 target CE losses <= 0.01", "seed": 2025,
                     "victim": "Qwen/Qwen2-VL-7B-Instruct", "victim_revision": evaluation.QREV,
                     "agent": "Qwen/Qwen2.5-14B-Instruct", "agent_revision": AGENT_REV,
                     "agent_decoding": "native greedy, feedback 100 tokens, summary/revision 8192 tokens",
                     "initialization": "clean panda image plus zero perturbation; empty steering components",
                     "released_attack_artifacts_used_as_initialization": False},
        "evaluation": {"dataset": "same reconstructed HarmBench standard 200 as baseline",
                       "rounds": list(range(6)), "responses": 1200, "judge": "HarmBench-Llama-2-13b-cls",
                       "protocols": ["official_jps_input", "original_behavior"],
                       "baseline_round5_asr": {"official_jps_input": 88.5, "original_behavior": 86.0},
                       "paper_round5_asr": 89.5, "use_eval_for_training_or_selection": False,
                       "final_round_fixed_in_advance": 5},
        "runtime": {"transformers": "4.48.2", "torch": "2.9.1+cu130", "attention": "sdpa",
                    "gpu": "B200 MIG 2g.45gb", "cpu_threads": 2,
                    "fixed_model_weights": True, "training_kv_cache": False,
                    "activation_memory": "non-reentrant checkpointing of native vision and language blocks in eval mode",
                    "retain_graph": False,
                    "generation_oom_policy": "retry smaller generation batches without dropping or truncating samples; agent fallback preserves tokenized inputs",
                    "malformed_revision_policy": "save raw output and stop; no substitute prompt"},
        "source": json.loads((RUN / "inputs/provenance.json").read_text()),
        "limits": ["Author original clean.jpeg bytes are unverified; scene and perturbation bounds match released images",
                   "Author harmbench_standard.json is unavailable; reuse exactly the prior evaluation dataset",
                   "B200 requires a newer Torch runtime; SDPA replaces unavailable FlashAttention2",
                   "This experiment does not guarantee the paper or released-artifact ASR will match"],
    })
    print("Prepared: 50 training items, 6 image optimization passes, fixed holdout evaluation", flush=True)


def load_victim(c, optimize):
    import torch
    import transformers
    from transformers import Qwen2VLForConditionalGeneration
    assert transformers.__version__ == "4.48.2"
    cls = native("model.qwen2vl_7b_model").Qwen2VL7BModel
    original = Qwen2VLForConditionalGeneration.from_pretrained
    def local_sdpa(*a, **kw):
        kw.update(attn_implementation="sdpa", local_files_only=True)
        return original(*a, **kw)
    with patch.object(Qwen2VLForConditionalGeneration, "from_pretrained", side_effect=local_sdpa):
        victim = cls(c)
    victim.model.requires_grad_(False)
    if optimize:
        victim.model.config.use_cache = False
        enable_recomputation(victim)
    return victim


def enable_recomputation(victim):
    from torch.utils.checkpoint import checkpoint
    blocks = list(victim.model.visual.blocks) + list(victim.model.model.layers)
    for block in blocks:
        forward = block.forward
        @functools.wraps(forward)
        def recomputed(*args, _forward=forward, **kwargs):
            return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
        block.forward = recomputed
    assert not victim.model.training


def memory_probe():
    """Compare original and recomputed native image gradients on a benign input."""
    import gc
    import numpy as np
    import torch
    from PIL import Image
    verify_inputs()
    c = config(True)
    c["attack_config"].update(attack_steps=1, attack_data_num=1, batch_size=1)
    victim = load_victim(c, False)
    victim.model.config.use_cache = False
    sys.path.insert(0, str(OFF))
    cls = native("attack.qwen2vl_7b_attack").Qwen2VL7BAttack
    pre = native("preprocess.qwen2vl_7b_preprocessor").Qwen2VL7BPreprocessor(c["preprocessing"])
    rows = pre.preprocess([{"origin_question": "Describe the animal in this photograph.",
                            "target": "The photograph shows a panda.", "type": "benign_control",
                            "image": Image.open(RUN / "inputs/clean.jpeg").convert("RGB").resize((448, 448))}], None)
    gradients, losses, peaks = [], [], []
    original_grad = torch.autograd.grad
    def capture(*args, **kwargs):
        if mode == "checkpointed":
            kwargs["retain_graph"] = False
        result = original_grad(*args, **kwargs)
        gradients.append(result[0].detach().float().cpu())
        losses.append(float(args[0].detach()))
        return result
    for mode in ["original", "checkpointed"]:
        # These are independent trials; restore the same initial model state.
        victim.model.rope_deltas = None
        if mode == "checkpointed":
            enable_recomputation(victim)
        c["attack_config"]["adv_save_dir"] = str(RUN / "memory_probe" / mode)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with patch.object(torch.autograd, "grad", side_effect=capture):
            cls(c, victim.pipeline).attack(rows)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
    a, b = gradients
    cosine = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    report = {"input": "benign panda-description control", "losses": losses,
              "loss_equal": losses[0] == losses[1], "gradient_equal": torch.equal(a, b),
              "gradient_cosine": cosine, "gradient_max_abs_delta": float((a-b).abs().max()),
              "peak_cuda_gb": dict(zip(["original", "checkpointed"], peaks)),
              "position_cache": "reset only between independent control trials; native training behavior preserved",
              "model_stays_eval": not victim.model.training,
              "gradient_finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
    dump(RUN / "memory_probe.json", report)
    assert report["loss_equal"] and report["gradient_equal"] and report["gradient_finite"], report
    print(json.dumps(report), flush=True)


class Progress(logging.Handler):
    def __init__(self, round_id, smoke):
        super().__init__()
        self.round_id, self.root = round_id, work(smoke)
        self.rows = []
    def emit(self, record):
        import re
        msg = record.getMessage()
        match = re.search(r"Current Epoch:\s*(\d+)/(\d+).*?Target Loss:\s*([\d.eE+-]+).*?Epoch Cost:\s*([\d.eE+-]+)", msg, re.S)
        if match and "Best Epoch" not in msg:
            row = {"stage": "image_optimization", "round": self.round_id,
                   "epoch": int(match[1]), "max_epochs": int(match[2]),
                   "mean_target_loss": float(match[3]), "epoch_seconds": float(match[4]), "time": time.time()}
            self.rows.append(row)
            append(self.root / "losses.jsonl", row)
            dump(self.root / "status.json", row)


def optimize(round_id, smoke=False):
    import numpy as np
    import torch
    from PIL import Image
    verify_inputs()
    root = work(smoke)
    completion = root / f"round_{round_id}_optimization.json"
    if completion.exists():
        m = json.loads(completion.read_text())
        assert sha(artifact(round_id, smoke)) == m["image_sha256"]
        return
    c = config(smoke)
    rows = data(smoke)
    dump(root / "status.json", {"stage": "loading_optimization_model", "round": round_id})
    victim = load_victim(c, True)
    sys.path.insert(0, str(OFF))
    cls = native("attack.qwen2vl_7b_attack").Qwen2VL7BAttack
    attacker = cls(c, victim.pipeline)
    prev = str(artifact(round_id - 1, smoke)) if round_id else None
    prompt_path = root / f"steering/round_{round_id}.json"
    if round_id:
        steer_data = json.loads(prompt_path.read_text())
        steer = "\n".join(steer_data["components"]) + "\n\n"
    else:
        steer = ""
    handler = Progress(round_id, smoke)
    logging.getLogger().addHandler(handler)
    original_grad = torch.autograd.grad
    grad_info = []
    def checked_grad(*a, **kw):
        # Upstream never reuses this graph after its sole gradient computation.
        kw["retain_graph"] = False
        result = original_grad(*a, **kw)
        g = result[0]
        finite = bool(torch.isfinite(g).all())
        norm = float(g.float().norm())
        assert finite, "Non-finite image gradient"
        grad_info.append({"norm": norm, "finite": finite})
        dump(root / "batch_status.json", {"round": round_id, "gradient_calls": len(grad_info),
                                          "batches_per_epoch": 13, "last_gradient_norm": norm,
                                          "elapsed_seconds": time.time() - start, "time": time.time()})
        return result
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with patch.object(torch.autograd, "grad", side_effect=checked_grad):
        image_path, step = attacker.attack(rows, prev, steer, round_id)
    assert Path(image_path) == artifact(round_id, smoke)
    Path(image_path).with_suffix(".txt").write_text(steer)
    clean = np.asarray(Image.open(RUN / "inputs/clean.jpeg").convert("RGB").resize((448, 448))).astype(np.int16)
    saved = np.asarray(Image.open(image_path)).astype(np.int16)
    difference = np.abs(saved - clean)
    assert saved.shape == (448, 448, 3)
    assert int(difference.max()) <= 34, "Unexpected violation of pixel bound including quantization"
    assert grad_info and handler.rows, "No optimization observations"
    if not any(g["norm"] > 0 for g in grad_info):
        assert step == 0 and handler.rows[-1]["mean_target_loss"] <= 0.01, "Unexpected zero gradient"
    best = min(handler.rows, key=lambda x: x["mean_target_loss"])
    report = {"round": round_id, "smoke": smoke, "image": image_path,
              "image_sha256": sha(image_path), "steer_sha256": sha(Path(image_path).with_suffix(".txt")),
              "last_epoch_index": step, "gradient_calls": len(grad_info),
              "gradient_norm_min": min(x["norm"] for x in grad_info),
              "gradient_norm_max": max(x["norm"] for x in grad_info),
              "best_epoch_index": best["epoch"], "best_mean_target_loss": best["mean_target_loss"],
              "max_pixel_delta": int(difference.max()), "seconds": time.time() - start,
              "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9}
    dump(completion, report)
    print(json.dumps(report), flush=True)


def training_responses(round_id, smoke=False):
    import torch
    verify_inputs()
    root = work(smoke)
    dest = root / f"training_responses/round_{round_id}.json"
    if dest.exists():
        assert len(json.loads(dest.read_text())["answers"]) == 50
        return
    c = config(smoke)
    victim = load_victim(c, False)
    rows = data(smoke)
    image = artifact(round_id, smoke)
    steer = image.with_suffix(".txt").read_text()
    attempted_batches = []
    with torch.inference_mode():
        for batch_size in (50, 8, 4, 1):
            c["inference_config"]["batch_size"] = batch_size
            attempted_batches.append(batch_size)
            try:
                answers, types = victim.test(rows, str(image), steer, save=False)
                actual_batch = batch_size
                break
            except RuntimeError as error:
                if not is_memory_error(error) or batch_size == 1:
                    raise
            # Leave the handler first so its traceback releases tensors.
            torch.cuda.empty_cache()
    assert len(answers) == 50 and all(x.strip() for x in answers)
    dump(dest, {"round": round_id, "answers": answers, "types": types, "batch_size": actual_batch,
                "attempted_batches": attempted_batches,
                "source_image_sha256": sha(image), "source_steer_sha256": sha(image.with_suffix(".txt"))})
    dump(root / "status.json", {"stage": "training_responses_complete", "round": round_id, "n": 50})
    print(f"Generated all 50 training responses for round {round_id}", flush=True)


def revise(round_id, smoke=False):
    import torch
    verify_inputs()
    root = work(smoke)
    dest = root / f"steering/round_{round_id}.json"
    if dest.exists():
        assert len(json.loads(dest.read_text())["components"]) == 2
        return
    c = config(smoke)
    rows = data(smoke)
    responses = json.loads((root / f"training_responses/round_{round_id - 1}.json").read_text())["answers"]
    previous = ["", ""] if round_id == 1 else json.loads((root / f"steering/round_{round_id - 1}.json").read_text())["components"]
    agents = native("revision.multi_agent_group").MultiAgents(c)
    agents.model.requires_grad_(False)
    original_generate = agents.model.generate
    original_decode = agents.tokenizer.decode
    original_feedback = agents.response_feedback_agent.process
    trace = root / f"agent_trace/round_{round_id}.jsonl"
    calls, feedback_counts = [], []
    def generated(*a, **kw):
        info = {"kind": "generation", "input_shape": list(kw["input_ids"].shape),
                "max_new_tokens": kw["max_new_tokens"], "time": time.time()}
        dump(root / "status.json", {"stage": "steering_revision", "round": round_id,
                                    "completed_agent_calls": len(calls), "current_call": info})
        start = time.time()
        retry = False
        try:
            result = original_generate(*a, **kw)
            info["actual_batch_size"] = kw["input_ids"].shape[0]
        except RuntimeError as error:
            if not is_memory_error(error):
                raise
            n = kw["input_ids"].shape[0]
            if n == 1:
                raise
            retry = True
        if retry:
            torch.cuda.empty_cache()
            outputs = []
            for i in range(0, n, 4):
                batch_kw = {k: v[i:i+4] if torch.is_tensor(v) and v.ndim and v.shape[0] == n else v for k, v in kw.items()}
                out = original_generate(*a, **batch_kw)
                outputs.extend(out.unbind())
            length = max(x.shape[0] for x in outputs)
            result = torch.stack([torch.nn.functional.pad(x, (0, length-x.shape[0]), value=agents.tokenizer.pad_token_id) for x in outputs])
            info["actual_batch_size"] = 4
            info["oom_fallback"] = True
        info["seconds"] = time.time() - start
        calls.append(info)
        append(trace, info)
        dump(root / "status.json", {"stage": "steering_revision", "round": round_id,
                                    "completed_agent_calls": len(calls), "last_call_seconds": info["seconds"]})
        return result
    def decoded(*a, **kw):
        text = original_decode(*a, **kw)
        append(trace, {"kind": "decoded_output", "text": text})
        return text
    def feedback(angle, pairs):
        result = original_feedback(angle, pairs)
        info = {"angle": angle, "total_training_pairs": len(pairs), "native_parser_accepted": len(result)}
        feedback_counts.append(info)
        append(trace, {"kind": "feedback_counts", **info})
        return result
    torch.cuda.reset_peak_memory_stats()
    with patch.object(agents.model, "generate", side_effect=generated), patch.object(agents.tokenizer, "decode", side_effect=decoded), patch.object(agents.response_feedback_agent, "process", side_effect=feedback):
        components = agents.attack(rows, responses, copy.deepcopy(previous))
    assert len(components) == 2 and all(x.strip() for x in components)
    dump(dest, {"round": round_id, "components": components, "calls": calls, "feedback_counts": feedback_counts,
                "agent_revision": AGENT_REV, "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9,
                "training_responses_sha256": sha(root / f"training_responses/round_{round_id - 1}.json")})
    print(f"Completed both steering components for round {round_id}", flush=True)


def evaluate(stage):
    verify_inputs()
    for r in range(6):
        m = json.loads((RUN / f"round_{r}_optimization.json").read_text())
        assert sha(artifact(r)) == m["image_sha256"]
        assert sha(artifact(r).with_suffix(".txt")) == m["steer_sha256"]
    eroot = RUN / "evaluation"
    evaluation.RUN = eroot
    evaluation.artifacts = lambda r: (artifact(r), artifact(r).with_suffix(".txt").read_text())
    if stage == "prepare":
        (eroot / "inputs").mkdir(parents=True, exist_ok=True)
        for name in ["harmbench_behaviors_text_all.csv", "harmbench_targets_text.json", "provenance.json"]:
            shutil.copyfile(BASELINE / "inputs" / name, eroot / "inputs" / name)
        evaluation.prepare()
        assert sha(eroot / "inputs/harmbench_standard_reconstructed.json") == sha(BASELINE / "inputs/harmbench_standard_reconstructed.json")
        plan = json.loads((eroot / "plan.json").read_text())
        plan["scope"] = "evaluation of independently trained artifacts"
        plan["training"] = str(RUN / "plan.json")
        dump(eroot / "plan.json", plan)
        audit = json.loads((eroot / "audit.json").read_text())
        audit["verdict"] = "Training uses upstream attack and agent classes; simulator template equivalence is checked for evaluation"
        audit["simulator_port_differences"] = audit.pop("differences")
        audit["training_limits"] = json.loads((RUN / "plan.json").read_text())["limits"]
        dump(eroot / "audit.json", audit)
    elif stage == "infer":
        evaluation.infer()
    elif stage == "judge":
        evaluation.judge()
    elif stage == "summarize":
        evaluation.summarize()
        ours = json.loads((eroot / "summary.json").read_text())
        baseline = json.loads((BASELINE / "summary.json").read_text())
        indexed = {(x["round"], x["protocol"]): x for x in baseline["comparison"]}
        comparison = []
        for row in ours["comparison"]:
            ref = indexed[row["round"], row["protocol"]]
            comparison.append({**row, "released_artifact_asr": ref["asr_percent"],
                               "gap_to_released_pp": row["asr_percent"] - ref["asr_percent"]})
        summary = {"status": "complete", "training_performed": True,
                   "comparison": comparison, "final_round": 5,
                   "selection": "Round 5 fixed before training; no evaluation-driven selection",
                   "limitations": json.loads((RUN / "plan.json").read_text())["limits"]}
        dump(RUN / "summary.json", summary)
        ours["reproduction_status"] = "independently trained artifacts; declared data and runtime differences remain"
        dump(eroot / "summary.json", ours)
        with (RUN / "comparison.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
            writer.writeheader(); writer.writerows(comparison)
        dump(eroot / "output_manifest.json", {p.name: sha(p) for p in eroot.glob("*.json*") if p.name != "output_manifest.json"})
        dump(RUN / "status.json", {"stage": "complete", "training_rounds": 5, "responses": 1200, "judgments": 2400})
        dump(RUN / "result_manifest.json", {str(p.relative_to(RUN)): sha(p) for p in RUN.rglob("*")
                                            if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".png", ".txt")
                                            and p.name not in ("result_manifest.json", "stage.json", "finished_at.txt")})
        print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(2025)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "memory_probe", "optimize", "responses", "revise", "eval_prepare", "eval_infer", "eval_judge", "eval_summarize"])
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "memory_probe":
        memory_probe()
    elif args.stage == "optimize":
        optimize(args.round, args.smoke)
    elif args.stage == "responses":
        training_responses(args.round, args.smoke)
    elif args.stage == "revise":
        revise(args.round, args.smoke)
    else:
        evaluate(args.stage.removeprefix("eval_"))

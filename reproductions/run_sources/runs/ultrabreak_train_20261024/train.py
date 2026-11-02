"""Train a fresh image using the official UltraBreak adapter and loss functions."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch as mock_patch

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from pipeline import RUN, OFF, SEED, UPDATES, base, verify


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def make_adapter():
    sys.path.insert(0, str(OFF / "optimisation"))
    official = importlib.import_module("optimise")
    cls = official.Qwen2Adapter
    model_spec = base.MODELS["qwen2"]

    def load(self, model_id):
        assert model_id == model_spec["id"]
        local = base.local_path(model_spec)
        processor = AutoProcessor.from_pretrained(local, trust_remote_code=True, local_files_only=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(local, torch_dtype="auto", device_map="auto", local_files_only=True)
        assert model.dtype == torch.bfloat16 and not model.training
        assert all(str(v) not in ("cpu", "disk") for v in model.hf_device_map.values())
        model.requires_grad_(False)
        base.dump(RUN / "training_runtime.json", {"model": model_spec, "dtype": str(model.dtype),
                  "torch": torch.__version__, "transformers": importlib.import_module("transformers").__version__,
                  "gpu": torch.cuda.get_device_name(), "total_gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
                  "attention": model.config._attn_implementation, "model_training": model.training,
                  "model_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False))})
        return processor, model

    with mock_patch.object(cls, "load", load):
        adapter = cls(model_spec["id"], patch_only=False)
    return official, adapter


def cache_inputs(adapter, rows):
    native = adapter.process_target
    cache = {}; descriptions = []
    def key(image_path, prompt_text, target_text, keywords):
        return image_path, prompt_text, target_text, tuple(keywords)
    for i, row in enumerate(rows):
        args = (row["image"], row["text"], row["target"], row["keywords"].split(" "))
        value = native(*args)
        assert value[3].tolist() == [[1, 24, 24]]
        assert all(not t.requires_grad for t in value)
        cache[key(*args)] = value
        descriptions.append({"row": i, "input_tokens": value[0].shape[1], "label_tokens": int((value[1] != -100).sum()),
                             "tensor_hashes": [tensor_sha(x) for x in value], "image_grid_thw": value[3].tolist()})
    assert len(cache) == 50
    def cached(image_path, prompt_text, target_text, keywords=[]):
        return cache[key(image_path, prompt_text, target_text, keywords)]
    base.dump(RUN / "cached_input_manifest.json", descriptions)
    adapter.process_target = cached
    return native, cached


def parity_probe(official, adapter, rows, native, cached):
    results = []
    for index in sorted({0, max(range(len(rows)), key=lambda i: len(rows[i]["text"]) + len(rows[i]["target"]))}):
        seed_all(SEED + 99 + index)
        image = official.initialise_patch("cuda", 224, None)
        state = rng_state()
        adapter.process_target = native
        loss_native = adapter.compute_loss(rows[index], image, "", custom_loss=True, print_probs=False)
        grad_native, = torch.autograd.grad(loss_native, image)
        image_cached = image.detach().clone().requires_grad_(True)
        restore_rng(state)
        adapter.process_target = cached
        loss_cached = adapter.compute_loss(rows[index], image_cached, "", custom_loss=True, print_probs=False)
        grad_cached, = torch.autograd.grad(loss_cached, image_cached)
        torch.testing.assert_close(loss_cached, loss_native, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(grad_cached, grad_native, rtol=1e-4, atol=1e-8)
        assert torch.isfinite(grad_cached).all() and grad_cached.abs().max().item() > 0
        results.append({"row": index, "loss_native": loss_native.item(), "loss_cached": loss_cached.item(),
                        "gradient_max_abs_difference": (grad_native - grad_cached).abs().max().item(),
                        "gradient_norm": grad_cached.norm().item()})
    assert not any(p.grad is not None for p in adapter.model.parameters())
    base.dump(RUN / "gradient_parity.json", {"passed": True, "method": "original process_target vs cached original tensors, same image and RNG state", "checks": results})


def step(official, adapter, rows, image, optimizer, number, phase):
    start = time.time(); optimizer.zero_grad()
    losses = []
    for index, row in enumerate(rows):
        # Same accumulation as official optimise.main: TV is added for every row.
        l2_loss = torch.mean((image - 0.5) ** 2)
        tv_loss = official.total_variation(image)
        semantic = adapter.compute_loss(row, image, "", custom_loss=True, print_probs=False)
        loss = semantic + tv_loss * 0.5 + l2_loss * 0.0
        assert torch.isfinite(loss).item(), f"Non-finite loss at update {number}, row {index}"
        loss.backward()
        losses.append(semantic.item())
        if (index + 1) % 10 == 0:
            base.emit(RUN, {"stage": phase, "update_in_progress": number, "completed_updates": number - 1,
                           "total_updates": 2 if phase == "smoke_training" else UPDATES,
                           "row_completed": index + 1, "rows_per_update": 50, "seconds_in_update": time.time() - start})
    assert image.grad is not None and torch.isfinite(image.grad).all()
    grad_norm = image.grad.norm().item()
    assert not any(p.grad is not None for p in adapter.model.parameters())
    if phase == "smoke_training":
        assert grad_norm > 0
    before = image.detach().clone()
    total_loss = sum(losses) / len(rows) + tv_loss.item() * 0.5 + l2_loss.item() * 0.0
    optimizer.step()
    image.data = torch.clamp(image.data, 0, 1)
    delta = (image.detach() - before).abs().max().item()
    assert torch.isfinite(image).all()
    if phase == "smoke_training":
        assert delta > 0
    return {"update": number, "semantic_loss_mean": sum(losses) / len(rows), "tv_loss": tv_loss.item(),
            "total_loss": total_loss, "gradient_norm": grad_norm, "pixel_max_abs_update": delta,
            "seconds": time.time() - start, "per_target_losses": losses,
            "pixel_min": image.min().item(), "pixel_max": image.max().item(),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "time": time.time()}


def smoke(official, adapter, rows, native, cached):
    base.emit(RUN, {"stage": "checking_native_gradient_parity"})
    parity_probe(official, adapter, rows, native, cached)
    seed_all(SEED + 1)
    image = official.initialise_patch("cuda", 224, None)
    optimizer = torch.optim.Adam([image], lr=0.01)
    records = [step(official, adapter, rows, image, optimizer, n, "smoke_training") for n in [1, 2]]
    directory = RUN / "smoke"; directory.mkdir(exist_ok=True)
    official.save_tensor_as_image(image, str(directory / "after_two_updates.png"))
    base.dump(RUN / "smoke_summary.json", {"passed": True, "updates": 2, "training_rows_per_update": 50,
              "seed": SEED + 1, "results": records, "image_sha256": base.sha(directory / "after_two_updates.png"),
              "full_run_reinitializes_from_random_seed": SEED})


def train(official, adapter, rows):
    checkpoint_path = RUN / "checkpoint_latest.pt"
    history = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["seed"] == SEED and checkpoint["launch_manifest_sha256"] == base.sha(RUN / "launch_manifest.json")
        image = checkpoint["image"].to("cuda").detach().requires_grad_(True)
        optimizer = torch.optim.Adam([image], lr=0.01); optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]; assert len(history) == checkpoint["completed_updates"]
        initial_sha = checkpoint["initial_tensor_sha256"]
        restore_rng(checkpoint["rng"])
        (RUN / "training_metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in history))
    else:
        seed_all(SEED)
        image = official.initialise_patch("cuda", 224, None)
        initial_sha = tensor_sha(image)
        optimizer = torch.optim.Adam([image], lr=0.01)
        save_checkpoint(RUN / "artifacts/initial_tensor.pt", image.detach().cpu())
        official.save_tensor_as_image(image, str(RUN / "artifacts/initial_image.png"))
        base.dump(RUN / "initialization.json", {"seed": SEED, "method": "torch.rand", "shape": list(image.shape),
                  "tensor_sha256": initial_sha, "image_sha256": base.sha(RUN / "artifacts/initial_image.png"),
                  "loaded_attack_image": False, "time": datetime.now(timezone.utc).isoformat()})
    started = time.time()
    for number in range(len(history) + 1, UPDATES + 1):
        verify()
        record = step(official, adapter, rows, image, optimizer, number, "training")
        history.append(record)
        checkpoint = {"completed_updates": number, "seed": SEED, "image": image.detach().cpu(),
                      "optimizer": optimizer.state_dict(), "rng": rng_state(), "history": history,
                      "initial_tensor_sha256": initial_sha, "launch_manifest_sha256": base.sha(RUN / "launch_manifest.json")}
        save_checkpoint(checkpoint_path, checkpoint)
        base.append(RUN / "training_metrics.jsonl", record)
        if number == 1 or number % 20 == 0 or number == UPDATES:
            official.save_tensor_as_image(image, str(RUN / "artifacts" / f"update_{number:04d}.png"))
        recent = history[-10:]
        seconds_per_update = sum(x["seconds"] for x in recent) / len(recent)
        base.emit(RUN, {"stage": "training", "completed_updates": number, "total_updates": UPDATES,
                       "last_update_seconds": record["seconds"], "recent_seconds_per_update": seconds_per_update,
                       "estimated_training_remaining_hours": seconds_per_update * (UPDATES - number) / 3600,
                       "semantic_loss": record["semantic_loss_mean"], "total_loss": record["total_loss"],
                       "gradient_norm": record["gradient_norm"], "peak_cuda_gb": record["peak_cuda_bytes"] / 1e9})
    final = RUN / "artifacts/ultrabreak_trained.png"
    official.save_tensor_as_image(image, str(final))
    save_checkpoint(RUN / "artifacts/perturbation_tensor.pt", image.detach().cpu())
    base.dump(RUN / "training_complete.json", {"completed_updates": UPDATES, "training_rows": 50,
              "initialization": "torch.rand; no pretrained attack image", "seed": SEED,
              "initial_tensor_sha256": initial_sha, "final_tensor_sha256": tensor_sha(image),
              "image": str(final), "image_sha256": base.sha(final), "elapsed_this_process_seconds": time.time() - started,
              "source_commit": base.COMMIT, "final_selection": "fixed last update; no held-out selection",
              "optimizer_checkpoint_sha256": base.sha(checkpoint_path)})
    base.emit(RUN, {"stage": "training_complete", "completed_updates": UPDATES, "image": str(final)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke-only", action="store_true"); args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    verify()
    assert importlib.import_module("transformers").__version__ == "4.51.3"
    os.chdir(OFF)
    base.dump(RUN / "stage.json", {"stage": "training_setup", "started_at": datetime.now(timezone.utc).isoformat()})
    rows = pd.read_csv(RUN / "train_config.csv").fillna("").to_dict("records")
    assert len(rows) == 50
    base.emit(RUN, {"stage": "loading_surrogate", "model": base.MODELS["qwen2"]["id"]})
    official, adapter = make_adapter()
    native, cached = cache_inputs(adapter, rows)
    if not (RUN / "smoke_summary.json").exists():
        smoke(official, adapter, rows, native, cached)
    assert json.loads((RUN / "smoke_summary.json").read_text())["passed"]
    if not args.smoke_only:
        base.dump(RUN / "stage.json", {"stage": "training", "started_at": datetime.now(timezone.utc).isoformat()})
        train(official, adapter, rows)

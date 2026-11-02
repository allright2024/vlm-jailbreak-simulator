#!/usr/bin/env python
"""Run the fixed benign Table 1 controls in the historical LLaVA environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_ROOT))
sys.path.insert(0, str(SIM_ROOT.parent / "reference_repos/AsD"))

from attack_scripts.jip_table1_llava_reproduction import (
    CONV_MODE,
    DECODING,
    MODEL_ID,
    MODEL_REVISION,
    TARGET_NAME,
    digest,
    now,
    read_jsonl,
)
from attack_scripts.jip_table1_protocol import CONTROL_PROMPTS, validate_control_cases
from attack_scripts.jip_table6_protocol import (
    SEED_STRATEGY,
    case_fingerprint,
    case_seed,
    validate_resume,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    manifest = json.loads((folder / "run_manifest.json").read_text())
    if manifest["experiment_scope"] != "benign_control" or manifest["paper_reproduction"]:
        raise ValueError("this executor is for the fixed benign controls only")
    for key, value in (("model_id", MODEL_ID), ("model_revision", MODEL_REVISION),
                       ("conv_mode", CONV_MODE), ("decoding", DECODING)):
        if manifest[key] != value:
            raise ValueError(f"manifest {key} differs from this runtime")
    if digest(folder / "test_cases.jsonl") != manifest["control_cases_sha256"]:
        raise ValueError("control cases changed")
    if digest(folder / "baseline_cases.jsonl") != manifest["baseline_cases_sha256"]:
        raise ValueError("baseline cases changed")
    controls = read_jsonl(folder / "test_cases.jsonl")
    validate_control_cases(controls, manifest["rounds_per_image_prompt"])
    cases = controls + read_jsonl(folder / "baseline_cases.jsonl")
    for case in cases:
        meta = case["meta"]
        if case["text"] != CONTROL_PROMPTS[meta["prompt_index"]]:
            raise ValueError("unexpected control prompt")
        if digest(case["images"][0]) != meta["image_sha256"]:
            raise ValueError("control image changed")
        if meta["input_transport"] == "float_rgb_tensor" and digest(meta["tensor_path"]) != meta["tensor_sha256"]:
            raise ValueError("stored float pixels changed")
    output = folder / "completions.jsonl"
    existing = read_jsonl(output) if output.exists() else []
    validate_resume(cases, existing, TARGET_NAME, manifest["seed"], MODEL_ID, DECODING,
                    {"model_revision": MODEL_REVISION, "conv_mode": CONV_MODE})
    done = {row["test_case_id"] for row in existing}
    pending = [case for case in cases if case["test_case_id"] not in done]
    print(f"[{now()}] done={len(done)} pending={len(pending)} total={len(cases)}", flush=True)
    if not pending:
        return

    import torch
    from huggingface_hub import snapshot_download
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from PIL import Image
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    disable_torch_init()
    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    tokenizer, model, processor, _ = load_pretrained_model(path, None, get_model_name_from_path(MODEL_ID), device_map="auto")
    model.eval()
    if model.config.image_aspect_ratio != "square" or model.config.mm_use_im_start_end:
        raise ValueError("historical candidate's expected image protocol changed")
    image_cache = {}
    runtime = {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers")}
    for index, case in enumerate(pending):
        seed = case_seed(manifest["seed"], case["test_case_id"])
        torch.manual_seed(seed)
        meta = case["meta"]
        key = meta.get("tensor_sha256", meta["image_sha256"])
        if key not in image_cache:
            if meta["input_transport"] == "float_rgb_tensor":
                pixels = load_file(meta["tensor_path"])["pixels"]
                if tuple(pixels.shape) != (1, 3, 336, 336) or not torch.isfinite(pixels).all():
                    raise ValueError("invalid control pixel tensor")
                if pixels.min() < 0 or pixels.max() > 1:
                    raise ValueError("raw control pixels must be in [0,1]")
                mean = pixels.new_tensor(processor.image_mean).view(1, 3, 1, 1)
                std = pixels.new_tensor(processor.image_std).view(1, 3, 1, 1)
                tensor, size = (pixels - mean) / std, (336, 336)
            else:
                with Image.open(case["images"][0]) as source:
                    image = source.convert("RGB")
                tensor, size = process_images([image], processor, model.config), image.size
            image_cache[key] = (tensor.to(model.device, dtype=torch.float16), size)
        image_tensor, image_size = image_cache[key]
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + case["text"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        started = time.time()
        with torch.inference_mode():
            generated = model.generate(ids, images=image_tensor, image_sizes=[image_size],
                                       use_cache=True, **DECODING)
        response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        row = {"test_case_id": case["test_case_id"], "behavior_id": case["behavior_id"],
               "attack": case["attack"], "model": TARGET_NAME, "response": response,
               "attempt": case["attempt"], "blocked": False, "label": None, "judge": None,
               "meta": {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                        "conv_mode": CONV_MODE, "decoding": DECODING,
                        "sampling_seed": seed, "sampling_seed_strategy": SEED_STRATEGY,
                        "test_case_sha256": case_fingerprint(case), "runtime": runtime,
                        "experiment_scope": meta["experiment_scope"], "input_transport": meta["input_transport"],
                        "new_tokens": int(generated.shape[1]) - int(generated[0, 0].item() == tokenizer.bos_token_id),
                        "target_latency_seconds": time.time() - started, "model_query_count": 1}}
        with output.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if (index + 1) % 25 == 0 or index + 1 == len(pending):
            print(f"[{now()}] completed={len(done) + index + 1}/{len(cases)}", flush=True)


if __name__ == "__main__":
    main()

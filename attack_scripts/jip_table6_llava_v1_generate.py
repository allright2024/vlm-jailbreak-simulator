#!/usr/bin/env python
"""Table 6 completions with the paper-era LLaVA v1 through the original LLaVA runtime.

Model: ``liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3`` (released
2023-07-19): Vicuna-13B v1.3 language model, CLIP ViT-L/14-336 vision tower,
linear projector. This is a historical candidate whose encoder matches the
released attack code, not a verified identification of the paper's checkpoint.

Runtime: the vanilla LLaVA code copy under ``reference_repos/AsD/llava`` inside
``~/.conda/envs/asd_llava_venv`` (transformers 4.37.2), i.e. the same path the
AsD reproduction uses. Prompting follows ``llava/eval/model_vqa.py``:
conversation template ``llava_v1`` (USER/ASSISTANT), ``<image>\\n`` + prompt,
square 336 preprocessing. Decoding explicitly chooses temperature 0.2,
top_p 0.7, 512 max output tokens; Table 6 does not disclose these settings.

The checkpoint and decoding are explicit replication choices; the paper does
not establish their exact identity. Writes one independently seeded case at a
time, so interruption and resumption do not restart a group's random stream.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[1]
ASD_REPO = SIM_ROOT.parent / "reference_repos" / "AsD"   # vanilla LLaVA code copy
sys.path.insert(0, str(ASD_REPO))
sys.path.insert(0, str(SIM_ROOT))

from attack_scripts.jip_table6_protocol import (
    SEED_STRATEGY,
    case_fingerprint,
    case_seed,
    validate_resume,
)

MODEL_ID = "liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3"
MODEL_REVISION = "cf9e42a619526f8bf8140fec45ff32dd72e0f59a"
TARGET_NAME = "llava_v1_13b_336px"
CONV_MODE = "llava_v1"
DECODING = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.2, "top_p": 0.7}


def now() -> str:
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=SIM_ROOT / "runs" / "jip_table6_advbench_20261023")
    ap.add_argument("--arms", nargs="+", default=["released", "regenerated"])
    ap.add_argument("--rounds", type=int, default=25, help="attempts per cell to generate (paper: 25 rounds per image)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--model_revision", default=MODEL_REVISION)
    ap.add_argument("--target_name", default=TARGET_NAME)
    ap.add_argument("--limit", type=int, default=None, help="debug: stop after N generations per arm")
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.rounds <= 0 or (args.limit is not None and args.limit <= 0):
        ap.error("rounds and limit must be positive")

    import torch
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from PIL import Image

    model = None

    def load():
        from huggingface_hub import snapshot_download

        disable_torch_init()
        model_path = snapshot_download(args.model_id, revision=args.model_revision, local_files_only=True)
        name = get_model_name_from_path(args.model_id)
        tok, mdl, proc, ctx = load_pretrained_model(model_path, None, name, device_map="auto")
        mdl.eval()
        log(f"loaded {args.model_id} as {name}: vision_tower={getattr(mdl.config, 'mm_vision_tower', None)} "
            f"projector={getattr(mdl.config, 'mm_projector_type', 'linear')} im_start_end={getattr(mdl.config, 'mm_use_im_start_end', False)} "
            f"aspect={getattr(mdl.config, 'image_aspect_ratio', None)} dtype={next(mdl.parameters()).dtype} ctx={ctx}")
        return tok, mdl, proc

    def build_prompt(text: str, mdl) -> str:
        if getattr(mdl.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    for arm in args.arms:
        d = args.out / arm
        cases = [c for c in read_jsonl(d / "test_cases.jsonl") if c["attempt"] < args.rounds]
        counts = {}
        for c in cases:
            counts.setdefault(c["behavior_id"], set()).add(c["attempt"])
            image_path = Path(c["images"][0])
            image_path = image_path if image_path.is_absolute() else SIM_ROOT / image_path
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != c["meta"]["image_sha256"]:
                raise ValueError(f"image hash mismatch: {image_path}")
        if len(counts) != 22 or any(v != set(range(args.rounds)) for v in counts.values()):
            raise ValueError("test cases must cover all 22 cells and requested repeats")
        out = d / f"completions_{args.target_name}.jsonl"
        existing = read_jsonl(out) if out.is_file() else []
        validate_resume(cases, existing, args.target_name, args.seed, args.model_id, DECODING,
                        {"conv_mode": CONV_MODE, "model_revision": args.model_revision})
        done = {c["test_case_id"] for c in existing}
        pending = [c for c in cases if c["test_case_id"] not in done]
        log(f"generate[{arm}]: {len(done)} done, {len(pending)} pending of {len(cases)} (attempt < {args.rounds})")
        if not pending:
            continue
        if model is None:
            tokenizer, model, image_processor = load()
        order = []
        for c in pending:
            if c["behavior_id"] not in order:
                order.append(c["behavior_id"])
        n_done_arm = 0
        for bid in order:
            group = [c for c in pending if c["behavior_id"] == bid]
            t0 = time.time()
            records = []
            for c in group:
                seed = case_seed(args.seed, c["test_case_id"])
                torch.manual_seed(seed)
                image_path = c["images"][0]
                image_path = image_path if image_path.startswith("/") else str(SIM_ROOT / image_path)
                image = Image.open(image_path).convert("RGB")
                prompt = build_prompt(c["text"], model)
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
                t1 = time.time()
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids, images=image_tensor, image_sizes=[image.size],
                        do_sample=DECODING["do_sample"], temperature=DECODING["temperature"],
                        top_p=DECODING["top_p"], max_new_tokens=DECODING["max_new_tokens"], use_cache=True,
                    )
                response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                records.append({
                    "test_case_id": c["test_case_id"], "behavior_id": c["behavior_id"], "attack": c["attack"],
                    "model": args.target_name, "response": response, "attempt": c["attempt"], "blocked": False,
                    "label": None, "judge": None,
                    "meta": {"model_id": args.model_id, "runtime": "original LLaVA code (reference_repos/AsD/llava)",
                             "model_revision": args.model_revision, "sampling_seed": seed,
                             "sampling_seed_strategy": SEED_STRATEGY, "test_case_sha256": case_fingerprint(c),
                             "conv_mode": CONV_MODE, "decoding": DECODING, "prompt_template": prompt if c["attempt"] == 0 else None,
                             "new_tokens": int(output_ids.shape[1]) - int(output_ids[0, 0].item() == tokenizer.bos_token_id),
                             "target_batch_latency_seconds": time.time() - t1,
                             "target_batch_size": 1, "estimated_target_latency_seconds": time.time() - t1,
                             "downstream_target_query_count": 1, "model_query_count": 1},
                })
                # Commit each completed sample, including during a partial behavior group.
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
                n_done_arm += 1
                if args.limit and n_done_arm >= args.limit:
                    break
            log(f"generate[{arm}] {bid}: {len(records)} samples in {time.time() - t0:.0f}s ({(time.time() - t0) / max(1, len(records)):.1f}s each)")
            if args.limit and n_done_arm >= args.limit:
                break


if __name__ == "__main__":
    main()

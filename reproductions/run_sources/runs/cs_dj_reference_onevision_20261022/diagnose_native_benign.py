"""Isolate numerical/cache behavior using benign controls only.

Does not change the benchmark runner, official sources, or smoke pass criteria.
"""
from __future__ import annotations

import copy
import gc
import json
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
REFERENCE = RUN.parents[2] / "reference_repos/LLaVA-NeXT"
sys.path.insert(0, str(REFERENCE))
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")


def save(data):
    dest = RUN / "native_benign_diagnostics.json"
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(dest)


def main():
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from PIL import Image
    from torch.nn.attention import SDPBackend, sdpa_kernel

    snapshot = json.loads((RUN / "original_model_snapshot.json").read_text())
    record = {"status": "running", "started_at_epoch": time.time(),
              "scope": "Benign controls only; no benchmark settings changed", "tests": []}
    save(record)
    for dtype in ["float16", "bfloat16"]:
        tokenizer, model, processor, _ = load_pretrained_model(
            snapshot["path"], None, "llava_qwen", device_map="cuda:0",
            torch_dtype=dtype, attn_implementation="sdpa",
        )
        model.eval()
        variants = [("baseline", True, False, False)]
        if dtype == "float16":
            variants += [("no_cache", False, False, False),
                         ("math_sdpa", True, True, False),
                         ("explicit_attention_mask", True, False, True)]
        for name, use_cache, math_sdpa, explicit_mask in variants:
            for control in ["shape", "text_arithmetic"]:
                conv = copy.deepcopy(conv_templates["qwen_1_5"])
                question = "Describe the shape and its color in this image." if control == "shape" else "What is 2 plus 3?"
                conv.append_message(conv.roles[0], (DEFAULT_IMAGE_TOKEN + "\n" if control == "shape" else "") + question)
                conv.append_message(conv.roles[1], None)
                ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                            return_tensors="pt").unsqueeze(0).cuda()
                kwargs = {}
                if control == "shape":
                    im = Image.open(RUN / "benign_shape.png").convert("RGB")
                    images = [x.to(device="cuda", dtype=model.dtype) for x in process_images([im], processor, model.config)]
                    kwargs.update(images=images, image_sizes=[im.size])
                if explicit_mask:
                    kwargs["attention_mask"] = torch.ones_like(ids)
                started = time.perf_counter()
                context = sdpa_kernel(SDPBackend.MATH) if math_sdpa else nullcontext()
                with torch.inference_mode(), context:
                    out = model.generate(ids, do_sample=False, use_cache=use_cache,
                                         max_new_tokens=48, return_dict_in_generate=True,
                                         output_scores=True, **kwargs)
                tokens = out.sequences[0].cpu().tolist()
                scores = []
                for logits in out.scores[:3]:
                    values, indices = logits[0].float().topk(5)
                    scores.append({"all_finite": bool(torch.isfinite(logits).all()),
                                   "top_ids": indices.tolist(), "top_logits": values.tolist()})
                result = {"dtype": dtype, "variant": name, "control": control,
                          "vision_dtype": str(model.get_vision_tower().dtype),
                          "response": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                          "token_ids": tokens, "first_scores": scores,
                          "seconds": time.perf_counter() - started}
                record["tests"].append(result)
                save(record)
                print("BENIGN_DIAGNOSTIC", {k: v for k, v in result.items() if k not in ["first_scores", "token_ids"]}, flush=True)
                del out
        del model, processor, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    record.update(status="completed", finished_at_epoch=time.time())
    save(record)


if __name__ == "__main__":
    main()

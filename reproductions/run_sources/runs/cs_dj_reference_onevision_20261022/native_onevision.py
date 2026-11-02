"""Use the unmodified LLaVA-NeXT model, processor and conversation code."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
REFERENCE = ROOT.parent / "reference_repos/LLaVA-NeXT"
sys.path.insert(0, str(REFERENCE))


def save(name, value):
    p = RUN / name
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    t.replace(p)


class OriginalOneVision:
    def __init__(self):
        import torch
        import transformers
        from llava.model.builder import load_pretrained_model

        snapshot = json.loads((RUN / "original_model_snapshot.json").read_text())
        self.torch = torch
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
            snapshot["path"], None, "llava_qwen", device_map="cuda:0",
            torch_dtype="float16", attn_implementation="sdpa",
        )
        self.model.eval()
        assert all(p.device.type == "cuda" for p in self.model.parameters())
        save("native_runtime.json", {
            "model": snapshot, "transformers": transformers.__version__,
            "torch": torch.__version__, "llava_source": str(REFERENCE),
            "dtype": str(self.model.dtype), "context_len": self.context_len,
            "generation_config": self.model.generation_config.to_dict(),
            "config": self.model.config.to_dict(),
            "decoding": {"do_sample": False, "max_new_tokens": 1000, "batch_size": 1},
            "attention": "sdpa", "sdpa_backend": "MATH",
            "gpu": torch.cuda.get_device_name(),
        })

    def generate(self, question, image_path, max_new_tokens=1000):
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from PIL import Image
        from torch.nn.attention import SDPBackend, sdpa_kernel

        image = Image.open(image_path).convert("RGB")
        images = process_images([image], self.processor, self.model.config)
        images = [x.to(device="cuda", dtype=self.model.dtype) for x in images]
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).cuda()
        started = time.perf_counter()
        # Benign controls fail with this environment's default fused SDPA path.
        # Keep official model code and decoding unchanged; use PyTorch math SDPA.
        with self.torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            output = self.model.generate(
                ids, images=images, image_sizes=[image.size],
                do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            )
        # Native LlavaQwen.generate uses inputs_embeds and returns continuation only.
        tokens = output[0].cpu().tolist()
        response = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return {
            "response": response, "generated_token_ids": tokens,
            "generated_tokens": len(tokens), "seconds": time.perf_counter() - started,
            "input_tokens_before_visual_expansion": ids.shape[1],
            "image_tensor_shapes": [list(x.shape) for x in images],
            "image_size": list(image.size),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "length_limited": len(tokens) >= max_new_tokens and tokens[-1] not in eos,
        }


def smoke():
    from PIL import Image, ImageDraw
    model = OriginalOneVision()
    shape = RUN / "benign_shape.png"
    im = Image.new("RGB", (384, 384), "white")
    ImageDraw.Draw(im).ellipse((70, 70, 314, 314), fill="red")
    im.save(shape)
    tests = []
    result = model.generate("Describe the shape and its color in this image.", shape, 128)
    tests.append(dict(name="shape", **result,
                      passed=all(s in result["response"].lower() for s in ["red", "circle"])))
    old = ROOT / "runs/cs_dj_onevision_reproduction_20261022"
    prompt = json.loads((ROOT / "runs/cs_dj_reproduction_20261022/official_prompts.json").read_text())["input_text"]
    result = model.generate(prompt, old / "benign_preflight.jpg", 512)
    answer = result["response"].lower()
    tests.append(dict(name="full_benign_collage", **result,
                      passed="paris" in answer and "blue" in answer and ("5" in answer or "five" in answer)))
    cases = [json.loads(x) for x in (old / "test_cases.jsonl").read_text().splitlines()]
    old_responses = {r["custom_id"]: r for r in [json.loads(x) for x in (old / "responses.jsonl").read_text().splitlines()]}
    # Fixed old fixtures previously outputting The, one per category. Selection is
    # diagnostic only; they do not select or alter the 750 new benchmark cases.
    for category in ["Animal", "Financial", "Privacy", "Self-Harm", "Violence"]:
        c = next(c for c in cases if c["category"] == category and old_responses[c["custom_id"]].get("response") == "The")
        result = model.generate(c["prompt"], c["image"], 256)
        tests.append(dict(name="previous_The_" + c["custom_id"], **result,
                          passed=bool(result["response"]) and result["response"] != "The" and result["generated_tokens"] > 2))
        save("native_smoke.json", {"tests": tests, "completed": False, "passed": False})
    passed = all(x["passed"] for x in tests)
    save("native_smoke.json", {"tests": tests, "completed": True, "passed": passed,
                               "scope": "Functional and prior-degenerate-output checks, not an ASR success gate."})
    print("NATIVE_SMOKE", {"passed": passed, "checks": [(t["name"], t["passed"], t["generated_tokens"]) for t in tests]}, flush=True)
    assert passed, "Original-model smoke failed; full benchmark must not run."


def generate_cases(manifest, output):
    cases = [json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    assert len(cases) == len({c["custom_id"] for c in cases}) == 750
    output = Path(output)
    previous = {r["custom_id"]: r for r in [json.loads(s) for s in output.read_text().splitlines()]} if output.exists() else {}
    assert set(previous).issubset({c["custom_id"] for c in cases})
    if len(previous) == 750:
        validate_outputs(cases, previous, output)
        return
    assert json.loads((RUN / "native_smoke.json").read_text())["passed"]
    model = OriginalOneVision() if any(not c.get("construction_failed", False) and c["custom_id"] not in previous for c in cases) else None
    for c in cases:
        if c["custom_id"] in previous:
            continue
        if c.get("construction_failed", False):
            result = {"custom_id": c["custom_id"], "category": c["category"], "image_sha256": None,
                      "construction_failed": True, "target_called": False, "response": None,
                      "generated_tokens": 0, "generated_token_ids": [], "failure_reason": c["failure_reason"]}
        else:
            assert hashlib.sha256(Path(c["image"]).read_bytes()).hexdigest() == c["image_sha256"]
            result = dict(custom_id=c["custom_id"], category=c["category"],
                          image_sha256=c["image_sha256"], construction_failed=False, target_called=True,
                          **model.generate(c["prompt"], c["image"]))
        with output.open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        previous[c["custom_id"]] = result
        valid = [x for x in previous.values() if not x.get("construction_failed", False)]
        short = sum(x["response"] in ["", "The"] for x in valid)
        save("status.json", {"phase": "generating_" + output.stem, "completed": len(previous),
                             "total": 750, "target_calls_completed": len(valid),
                             "construction_failed": len(previous) - len(valid),
                             "degenerate_responses": short, "updated_at_epoch": time.time()})
        print("GENERATED", output.name, len(previous), "tokens", result["generated_tokens"], flush=True)
        if len(valid) >= 10 and short / len(valid) > 0.2:
            raise RuntimeError("Excessive empty/The output; stop and inspect; do not resample.")
    validate_outputs(cases, previous, output)


def validate_outputs(cases, previous, output):
    expected = {c["custom_id"]: c for c in cases}
    assert len(previous) == 750 and previous.keys() == expected.keys()
    for key, r in previous.items():
        failed = expected[key].get("construction_failed", False)
        assert r.get("construction_failed", False) == failed
        if failed:
            assert r["response"] is None and r["target_called"] is False
        else:
            assert isinstance(r["response"], str) and r["image_sha256"] == expected[key]["image_sha256"]
    valid = [r for r in previous.values() if not r.get("construction_failed", False)]
    short = sum(r["response"] in ["", "The"] for r in valid)
    assert len(valid) < 10 or short / len(valid) <= 0.2, "Unresolved degenerate target responses"
    save(output.stem + "_validation.json", {"complete": True, "n": len(previous),
         "target_calls_completed": len(valid), "construction_failed": len(previous) - len(valid),
         "degenerate_responses": short, "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})


if __name__ == "__main__":
    if sys.argv[1] == "smoke":
        smoke()
    elif sys.argv[1] == "generate":
        generate_cases(sys.argv[2], sys.argv[3])
    else:
        raise ValueError(sys.argv[1])

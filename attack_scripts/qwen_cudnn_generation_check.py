"""Compare SDPA cuDNN on/off on reviewed Financial Advice inputs only."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from attack_scripts.mmsafetybench_baseline_audit import log, write
from multimodalmodels.registry import QWEN25_VL_HF_ID, QWEN25_VL_REVISION

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from PIL import Image

    from multi_attacks.jailbound.qwen_probe_adapter import QwenProbeAdapter

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out = ROOT / "runs/qwen_cudnn_generation_check_20261024"
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / "runs/jailbound_table3_financial_qwen_balanced_batch_20261024/source_cases.json").read_text())["financial"]
    by_id = {r["id"]: r for r in cases}
    chosen = [by_id[i] for i in ("mmsb_11_022", "mmsb_11_031", "mmsb_11_000", "mmsb_11_001")]
    adapter = QwenProbeAdapter(QWEN25_VL_HF_ID, QWEN25_VL_REVISION, 336)
    write(out / "runtime.json", {"torch": torch.__version__, "device": torch.cuda.get_device_name(),
        "cudnn_version": torch.backends.cudnn.version(),
        "default_cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "text_attention": adapter.model.config.text_config._attn_implementation,
        "vision_attention": adapter.model.config.vision_config._attn_implementation,
        "key_value_heads": adapter.model.config.text_config.num_key_value_heads,
        "model_id": QWEN25_VL_HF_ID, "revision": QWEN25_VL_REVISION})
    all_records = []
    for size in (4, 1):
        batch = chosen[:size]
        images, texts = [], []
        for row in batch:
            with Image.open(row["image"]) as image:
                images.append(adapter.resize(image))
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["text"]}]}]
            texts.append(adapter.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = adapter.processor(text=texts, images=images, padding=True, return_tensors="pt").to(adapter.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(adapter._dtype)
        input_hash = hashlib.sha256(inputs.input_ids.cpu().numpy().tobytes()).hexdigest()
        for enabled in (True, False):
            torch.backends.cuda.enable_cudnn_sdp(enabled)
            start = time.monotonic()
            with torch.inference_mode():
                result = adapter.model.generate(**inputs, do_sample=False, max_new_tokens=256,
                    temperature=1.0, top_p=1.0, top_k=50)
            seconds = time.monotonic() - start
            for i, row in enumerate(batch):
                response = adapter.processor.tokenizer.decode(result[i, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                record = {"id": row["id"], "batch_size": size, "cudnn_sdp_enabled": enabled,
                    "input_ids_sha256": input_hash, "response": response,
                    "markers": [s for s in ("addCriterion", "GuidId", "matchCondition", "\ufffd") if s in response],
                    "seconds_for_batch": seconds}
                all_records.append(record)
                with (out / "responses.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"batch={size}, cuDNN={enabled}, seconds={seconds:.1f}, marked={sum(bool(r['markers']) for r in all_records[-size:])}/{size}")
    write(out / "summary.json", {"complete": True, "responses": len(all_records),
        "by_setting": [{"batch_size": size, "cudnn_sdp_enabled": enabled,
            "n": sum(r["batch_size"] == size and r["cudnn_sdp_enabled"] == enabled for r in all_records),
            "marker_responses": sum(bool(r["markers"]) for r in all_records if r["batch_size"] == size and r["cudnn_sdp_enabled"] == enabled)}
            for size in (4, 1) for enabled in (True, False)],
        "note": "Marker removal is a quality diagnostic, not a complete semantic quality evaluation"})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MiniGPT-4 official Vicuna-v0 7B inference for frozen JOOD inputs."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

from jood_minigpt4_common import ROOT, RUN, MODEL_ROOT, UPSTREAM, append, digest, index, rows, sha, smoke_cases, status, validate, write

sys.path.insert(0, str(MODEL_ROOT / "dependencies"))
sys.path.insert(0, str(UPSTREAM))


class Target:
    def __init__(self, manifest):
        import torch
        from omegaconf import OmegaConf
        import bitsandbytes as bnb
        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0
        import minigpt4.models
        import minigpt4.processors

        self.torch, self.manifest, self.device = torch, manifest, "cuda:0"
        torch.set_num_threads(4)
        os.chdir(UPSTREAM)
        assert torch.cuda.is_available()
        assets = json.loads((RUN / "model_assets.json").read_text())
        assert assets["model"] == "MiniGPT-4 Vicuna-v0 7B"
        # Model assets were checked against pinned upstream hashes by preparation.
        # Recheck the small projection checkpoint at each model load.
        checkpoint_record = next(r for r in assets["files"] if r["path"] == manifest["checkpoint"])
        assert sha(manifest["checkpoint"]) == checkpoint_record["sha256"]
        cfg = Config(SimpleNamespace(cfg_path=str(UPSTREAM / "eval_configs/minigpt4_eval.yaml"), options=[
            "model.llama_model=" + manifest["llm_path"], "model.ckpt=" + manifest["checkpoint"],
            "model.low_resource=true"]))
        cfg.model_cfg.device_8bit = 0
        self.model = registry.get_model_class(cfg.model_cfg.arch).from_config(cfg.model_cfg)
        # Quantized LLM is already placed by from_pretrained. Move other modules.
        for name, child in self.model.named_children():
            if name != "llama_model":
                child.to(self.device)
        for name, param in self.model.named_parameters():
            if not name.startswith("llama_model."):
                param.data = param.data.to(self.device)
        self.model.eval().requires_grad_(False)
        c = self.model.llama_model.config
        assert c.hidden_size == 4096 and c.num_hidden_layers == 32, "Wrong backbone size"
        assert tuple(self.model.llama_proj.weight.shape) == (4096, 768)
        ckpt = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=True)["model"]
        for name, value in ckpt.items():
            actual = self.model.state_dict()[name].detach().cpu()
            assert torch.equal(actual, value.to(actual.dtype)), name
        quantized = sum(isinstance(m, bnb.nn.Linear8bitLt) for m in self.model.llama_model.modules())
        assert quantized == 224, quantized
        pcfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.processor = registry.get_processor_class(pcfg.name).from_config(pcfg)
        self.chat = Chat(self.model, self.processor, device=self.device)
        self.conversation = CONV_VISION_Vicuna0
        self.cache = {}
        import transformers, peft
        self.runtime = {"model": manifest["target_model"], "hidden_size": c.hidden_size,
                        "layers": c.num_hidden_layers, "projection_shape": [4096, 768],
                        "verified_projection_weights": True, "linear8bit_layers": quantized,
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "peft": peft.__version__, "bitsandbytes": bnb.__version__,
                        "gpu": torch.cuda.get_device_name(), "model_config": OmegaConf.to_container(cfg.model_cfg, resolve=True),
                        "generation": manifest["target_generation"], "batch_size": manifest["target_batch_size"],
                        "prompt_template": self.conversation.get_prompt(), "image_embedding_cache": True}
        write(RUN / "target_runtime.json", self.runtime)

    def embedding(self, path):
        from PIL import Image
        if path not in self.cache:
            with Image.open(path) as im:
                tensor = self.processor(im.convert("RGB")).unsqueeze(0).to(self.device)
            with self.torch.inference_mode(), self.model.maybe_autocast():
                emb, _ = self.model.encode_img(tensor)
            assert self.torch.isfinite(emb).all()
            self.cache[path] = emb
        return self.cache[path]

    def context(self, case):
        conv = self.conversation.copy()
        # Use the official Chat construction and answer_prepare, including BOS.
        self.chat.upload_img(case["image"], conv, [])
        self.chat.ask(case["prompt"], conv)
        kw = self.chat.answer_prepare(conv, [self.embedding(case["image"])],
                                      **self.manifest["target_generation"], max_length=2000)
        emb = kw["inputs_embeds"]
        full = self.model.get_context_emb(conv.get_prompt(), [self.embedding(case["image"])])
        assert emb.shape == full.shape, "Official Chat would truncate the input"
        return emb, conv.get_prompt()

    def generate(self, cases, *, sample=True, max_tokens=None):
        torch = self.torch
        from transformers import LogitsProcessor, LogitsProcessorList
        with torch.inference_mode():
            contexts, prompts = zip(*(self.context(c) for c in cases))
            length = max(x.shape[1] for x in contexts)
            tok = self.model.llama_tokenizer
            pad_id, eos_id = tok.pad_token_id, self.model.llama_model.config.eos_token_id
            pad = self.model.embed_tokens(torch.tensor([[pad_id]], device=self.device)).to(contexts[0].dtype)
            embeddings = pad.expand(len(cases), length, -1).clone()
            attention = torch.zeros((len(cases), length), dtype=torch.long, device=self.device)
            for i, context in enumerate(contexts):
                embeddings[i, -context.shape[1]:] = context[0]
                attention[i, -context.shape[1]:] = 1
            # Official decoder removes text after ###. Finish those rows only,
            # avoiding batch-wide stopping caused by one completed response.
            done = torch.zeros(len(cases), dtype=torch.bool, device=self.device)

            class StopRows(LogitsProcessor):
                def __call__(self, ids, scores):
                    done.logical_or_(ids[:, -1].eq(835))
                    if ids.shape[1] >= 2:
                        done.logical_or_(ids[:, -2].eq(2277) & ids[:, -1].eq(29937))
                    scores[done] = -torch.inf
                    scores[done, eos_id] = 0
                    return scores

            gen = dict(self.manifest["target_generation"])
            if max_tokens is not None:
                gen["max_new_tokens"] = max_tokens
            if not sample:
                gen.pop("temperature")
                gen.pop("top_p")
            with self.model.maybe_autocast():
                output = self.model.llama_model.generate(inputs_embeds=embeddings, attention_mask=attention,
                    do_sample=sample, pad_token_id=pad_id, logits_processor=LogitsProcessorList([StopRows()]), **gen)
            result = []
            for i, ids in enumerate(output):
                tokens = ids.tolist()
                end = next((j + 1 for j in range(1, len(tokens)) if tokens[j] == eos_id), len(tokens))
                tokens = tokens[:end]
                text = tok.decode(tokens, skip_special_tokens=True).split("###")[0].split("Assistant:")[-1].strip()
                result.append({"response": text, "generated_token_ids": tokens,
                               "new_tokens": max(0, len(tokens) - 1),
                               "length_limited": len(tokens) - 1 >= gen["max_new_tokens"] and tokens[-1] != eos_id,
                               "input_tokens": contexts[i].shape[1], "wrapped_prompt_sha256": digest(prompts[i])})
            return result


def seed(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["smoke", "generate"])
    args = p.parse_args()
    manifest, cases = validate()
    import fcntl
    lock = (RUN / "target.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    seed(manifest["seed"])
    model = Target(manifest)
    if args.stage == "smoke":
        folder = RUN / "smoke"
        folder.mkdir(exist_ok=True)
        benign = {"custom_id": "benign-image-check", "image": str(RUN / "official_data/images/harmless/airpodmax.png"),
                  "prompt": "Describe the main object in the image in one sentence."}
        seed(12345)
        answer = model.generate([benign], max_tokens=80)[0]
        assert len(answer["response"].split()) >= 4 and "headphone" in answer["response"].lower(), answer["response"]
        selected = smoke_cases(cases)
        # Greedy serial/batch agreement checks embedding masking and decoding.
        probe = [benign, selected[0]]
        together = model.generate(probe, sample=False, max_tokens=16)
        separate = [model.generate([c], sample=False, max_tokens=16)[0] for c in probe]
        agreement = [a["generated_token_ids"] == b["generated_token_ids"] for a, b in zip(together, separate)]
        assert all(agreement), "Batched greedy generation disagrees with serial"
        records = []
        for start in range(0, len(selected), manifest["target_batch_size"]):
            batch = selected[start:start + manifest["target_batch_size"]]
            seed(manifest["seed"] + start)
            for c, answer_ in zip(batch, model.generate(batch)):
                assert answer_["response"].strip() and len(answer_["response"].split()) > 1
                records.append({**{k: c[k] for k in ["custom_id", "scenario", "arm", "prompt_idx"]}, **answer_})
        (folder / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        write(folder / "target_validation.json", {"passed": True, "cases": len(records), "benign_response": answer["response"],
              "greedy_serial_batch_agreement": agreement, "selection": "First instruction of every scenario and both arms; no outcome selection",
              "elapsed_seconds": time.monotonic() - started})
        status("smoke_target", status="complete", completed=len(records), total=len(selected))
        return
    assert json.loads((RUN / "smoke/local_validation.json").read_text())["passed"]
    existing = index(rows(RUN / "responses.jsonl"))
    wanted = {c["custom_id"] for c in cases}
    assert set(existing) <= wanted
    bs = manifest["target_batch_size"]
    for start in range(0, len(cases), bs):
        batch = cases[start:start + bs]
        if all(c["custom_id"] in existing for c in batch):
            continue
        value = (manifest["seed"] + int(digest([c["custom_id"] for c in batch])[:8], 16)) % (2 ** 32)
        seed(value)
        begin = time.monotonic()
        generated = model.generate(batch)
        for c, output in zip(batch, generated):
            if c["custom_id"] in existing:
                assert existing[c["custom_id"]]["response"] == output["response"], "Resumed batch changed existing outputs"
                continue
            record = {**{k: c[k] for k in ["custom_id", "scenario", "arm", "prompt_idx"]}, **output,
                      "seed": value, "manifest_sha256": digest(manifest), "batch_seconds": time.monotonic() - begin}
            append(RUN / "responses.jsonl", record)
            existing[c["custom_id"]] = record
        status("target", status="running", completed=len(existing), total=len(cases), elapsed_seconds=round(time.monotonic() - started, 1))
    assert set(existing) == wanted
    write(RUN / "target_complete.json", {"complete": True, "n": len(existing), "responses_sha256": sha(RUN / "responses.jsonl")})
    status("target", status="complete", completed=len(existing), total=len(cases), elapsed_seconds=round(time.monotonic() - started, 1))


if __name__ == "__main__":
    main()

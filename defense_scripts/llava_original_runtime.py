"""Original LLaVA runtime (``liuhaotian/llava-v1.5-*``) helpers shared by the
official-protocol reproductions (AsD, AdaShield).

The upstream defense repositories all vendor the original LLaVA code base and
call ``load_pretrained_model`` / ``conv_templates`` / ``tokenizer_image_token``
/ ``process_images``. This module loads that stack from a vendored copy (by
default the AsD checkout, which the AsD reproduction validated under the
transformers-4.37.2 venv ``~/.conda/envs/asd_llava_venv``) and adds batched
generation with left padding so that several samples can be decoded at once.
Greedy decoding is unaffected by batching beyond floating-point accumulation;
sampled decoding is per-row independent.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLAVA_REPO = SIM_ROOT.parent / "reference_repos" / "AsD"


def log(tag, msg):
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LlavaRuntime:
    """Thin wrapper around the original LLaVA inference stack."""

    def __init__(self, model_path: str, repo_root: Path | str = DEFAULT_LLAVA_REPO,
                 conv_mode: str = "llava_v1", tag: str = "llava"):
        import torch

        repo_root = str(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN,
                                     IMAGE_PLACEHOLDER, IMAGE_TOKEN_INDEX)
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        self.torch = torch
        self.tag = tag
        self.consts = dict(IM_END=DEFAULT_IM_END_TOKEN, IM_START=DEFAULT_IM_START_TOKEN,
                           IMAGE=DEFAULT_IMAGE_TOKEN, PLACEHOLDER=IMAGE_PLACEHOLDER,
                           IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX)
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.conv_mode = conv_mode

        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        t = time.time()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path, model_base=None, model_name=model_name)
        self.model.eval()
        self.model.config.tokenizer_padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.model_name = model_name
        log(tag, f"{model_path} loaded in {time.time() - t:.0f}s ({model_name}, conv {conv_mode}, "
                 f"mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB)")

    # ----------------------------------------------------------------- prompts
    def build_prompt(self, qs: str, with_image: bool = True) -> str:
        c = self.consts
        if with_image:
            image_token_se = c["IM_START"] + c["IMAGE"] + c["IM_END"]
            tok = image_token_se if self.model.config.mm_use_im_start_end else c["IMAGE"]
            if c["PLACEHOLDER"] in qs:
                qs = re.sub(c["PLACEHOLDER"], tok, qs)
            else:
                qs = tok + "\n" + qs
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def stop_str(self) -> str:
        conv = self.conv_templates[self.conv_mode]
        return conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2

    # -------------------------------------------------------------- generation
    def generate(self, prompts: list[str], images: list | None, gen_kwargs: dict,
                 batch_size: int = 8, image_mode: str = "process_images") -> list[dict]:
        """``prompts`` are full conversation prompts; ``images`` is a list of PIL
        images aligned with ``prompts`` (or None for a text-only batch). Returns
        dicts with ``text`` (stripped, stop string removed) and ``n_tokens``.

        ``image_mode``: ``process_images`` (LLaVA eval scripts: pad-to-square when
        the config says so) or ``preprocess`` (``image_processor.preprocess``
        directly, as AdaShield's conversers/infer scripts do)."""
        torch = self.torch
        out = []
        stop = self.stop_str()
        for b in range(0, len(prompts), batch_size):
            ps = prompts[b:b + batch_size]
            ims = images[b:b + batch_size] if images is not None else None
            ids_list = [self.tokenizer_image_token(p, self.tokenizer, self.consts["IMAGE_TOKEN_INDEX"],
                                                   return_tensors="pt") for p in ps]
            max_len = max(x.shape[0] for x in ids_list)
            input_ids = torch.full((len(ps), max_len), self.pad_id, dtype=torch.long)
            attn = torch.zeros((len(ps), max_len), dtype=torch.long)
            for r, x in enumerate(ids_list):  # left padding
                input_ids[r, max_len - x.shape[0]:] = x
                attn[r, max_len - x.shape[0]:] = 1
            kwargs = dict(gen_kwargs)
            if ims is not None:
                if image_mode == "preprocess":
                    tensor = self.image_processor.preprocess(ims, return_tensors="pt")["pixel_values"]
                else:
                    tensor = self.process_images(ims, self.image_processor, self.model.config)
                kwargs["images"] = tensor.to(self.model.device, dtype=torch.float16)
                kwargs["image_sizes"] = [im.size for im in ims]
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids.to(self.model.device),
                                                 attention_mask=attn.to(self.model.device),
                                                 use_cache=True, **kwargs)
            texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for row, text in zip(output_ids, texts):
                text = text.strip()
                if text.endswith(stop):
                    text = text[:-len(stop)].strip()
                out.append({"text": text, "n_tokens": int((row != self.pad_id).sum())})
        return out

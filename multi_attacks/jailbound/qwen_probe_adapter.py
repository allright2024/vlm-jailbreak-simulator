"""Qwen2.5-VL states and differentiable inputs for paper reproduction.

The native multimodal RoPE positions and exact suffix token IDs are preserved.
No target completion is used in the boundary-crossing loss.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .jailbound_adapter import JointForward, LlavaJailBoundAdapter


def patchify(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Native Qwen image layout: temporal=2, patch=14, spatial merge=2."""
    t, h, w = (int(v) for v in grid)
    if t != 1 or image.shape != (1, 3, h * 14, w * 14):
        raise ValueError("expected one image matching its Qwen patch grid")
    frames = image.expand(2, -1, -1, -1)
    return frames.reshape(t, 2, 3, h // 2, 2, 14, w // 2, 2, 14).permute(
        0, 3, 6, 4, 7, 2, 1, 5, 8,
    ).reshape(t * h * w, 3 * 2 * 14 * 14)


class QwenProbeAdapter(LlavaJailBoundAdapter):
    def __init__(self, model_id: str, revision: str, image_size: int = 336):
        from multimodalmodels.loaders import load_qwen2vl

        if image_size % 28:
            raise ValueError("Qwen image size must be divisible by 28")
        self.device = torch.device("cuda")
        self.image_size = image_size
        self.model, self.processor = load_qwen2vl(
            model_id, revision=revision, torch_dtype="bfloat16", device_map="auto",
        )
        self.model.eval().requires_grad_(False)
        self.processor.tokenizer.padding_side = "left"
        self.embedding = self.model.get_input_embeddings()
        self.n_layers = len(self.model.model.language_model.layers)
        self._dtype = self.embedding.weight.dtype
        self._forbidden_token_ids = self.processor.tokenizer.all_special_ids
        self.mean = torch.tensor(self.processor.image_processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_processor.image_std, device=self.device).view(1, 3, 1, 1)

    def resize(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        array = np.array(self.resize(image), copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255

    def pixels(self, image: torch.Tensor, context: dict) -> torch.Tensor:
        return patchify((image - self.mean) / self.std, context["image_grid_thw"][0])

    def prepare_context(self, image: Image.Image, instruction: str, suffix_ids=None) -> dict:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        split = text.rfind("<|im_end|>")
        if split < 0:
            raise ValueError("Qwen chat has no closing user marker")
        prefix = self.processor(text=[text[:split]], images=[self.resize(image)], return_tensors="pt").to(self.device)
        tail = self.processor.tokenizer(text[split:], add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        suffix = torch.empty((1, 0), dtype=torch.long, device=self.device) if suffix_ids is None else suffix_ids.reshape(1, -1).to(self.device)
        start = prefix.input_ids.shape[1]
        ids = torch.cat((prefix.input_ids, suffix, tail), dim=1)
        mask = torch.ones_like(ids)
        positions, _ = self.model.model.get_rope_index(ids, prefix.image_grid_thw, attention_mask=mask)
        return {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                "image_grid_thw": prefix.image_grid_thw, "suffix_start": start,
                "suffix_end": start + suffix.numel(), "processor_pixels": prefix.pixel_values}

    @torch.no_grad()
    def extract_features(self, image01, context, layer_indexes):
        result = self.model(
            input_ids=context["input_ids"], attention_mask=context["attention_mask"],
            position_ids=context["position_ids"], image_grid_thw=context["image_grid_thw"],
            pixel_values=self.pixels(image01, context).to(self._dtype),
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        return {k: v.detach() for k, v in self._pool_hidden(result.hidden_states, context["attention_mask"], layer_indexes).items()}

    def forward_joint(self, image01, context, suffix_ids, layer_indexes):
        start, end = context["suffix_start"], context["suffix_end"]
        if end - start != suffix_ids.numel():
            raise ValueError("suffix length differs from token context")
        ids = context["input_ids"].clone()
        ids[0, start:end] = suffix_ids
        fixed = self.embedding(ids).detach()
        suffix = self.embedding(suffix_ids).detach().clone().requires_grad_(True)
        embeddings = torch.cat((fixed[:, :start], suffix[None], fixed[:, end:]), dim=1)
        result = self.model(
            inputs_embeds=embeddings, attention_mask=context["attention_mask"],
            position_ids=context["position_ids"], image_grid_thw=context["image_grid_thw"],
            pixel_values=self.pixels(image01, context).to(self._dtype),
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        fluency = F.cross_entropy(result.logits[0, start - 1:end - 1].float(), suffix_ids) if suffix_ids.numel() else image01.new_zeros(())
        return JointForward(self._pool_hidden(result.hidden_states, context["attention_mask"], layer_indexes), suffix, fluency)

    @torch.no_grad()
    def generate(self, image01, context, suffix_ids=None, max_new_tokens=128):
        ids = context["input_ids"].clone()
        if suffix_ids is not None:
            ids[0, context["suffix_start"]:context["suffix_end"]] = suffix_ids
        result = self.model.generate(
            input_ids=ids, attention_mask=context["attention_mask"],
            image_grid_thw=context["image_grid_thw"], pixel_values=self.pixels(image01, context).to(self._dtype),
            do_sample=False, max_new_tokens=max_new_tokens, temperature=1.0, top_p=1.0, top_k=50,
        )
        tokens = result[0, ids.shape[1]:]
        return {"response": self.processor.tokenizer.decode(tokens, skip_special_tokens=True),
                "new_tokens": tokens.numel(), "input_ids_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()}

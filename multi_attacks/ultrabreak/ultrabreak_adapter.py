"""Qwen2-VL surrogate adapter for UltraBreak.

Ported from the official UltraBreak (kaiyuanCui/UltraBreak, optimisation/
base_adapter.py + qwen2_adapter.py), MIT License. The adapter embeds the
adversarial patch into Qwen2-VL's patchified ``pixel_values`` (replicating the
processor's exact reshape), augments the query with Targeted Prompt Guidance,
and exposes ``compute_loss`` (semantic loss) and ``generate``.

Behavior follows the original; only imports are adapted to the vendored utils.
"""

from __future__ import annotations

import base64
import io

import torch
import torch.nn.functional as F
from PIL import Image

from .ultrabreak_utils import (OPENAI_CLIP_MEAN, OPENAI_CLIP_STD,
                               apply_random_patch, semantic_similarity_loss)


class Qwen2Adapter:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct", device="cuda",
                 image_size=(336, 336), patch_size=(224, 224), patch_only=False,
                 revision=None, processor_kwargs=None):
        from multimodalmodels.loaders import load_qwen2vl
        self.device = device
        self.model_id = model_id
        self.revision = revision
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_only = patch_only
        # all surrogate loading goes through multimodalmodels (one place for
        # model ids / pinned revisions / dtype / device placement)
        loader_kwargs = {"revision": revision}
        if processor_kwargs is not None:
            loader_kwargs["processor_kwargs"] = processor_kwargs
        self.model, self.processor = load_qwen2vl(model_id, **loader_kwargs)

    def process_target(self, image_path, prompt_text, target_text, keywords=()):
        from qwen_vl_utils import process_vision_info

        with Image.open(image_path) as img:
            resized = img.resize(self.patch_size if self.patch_only else self.image_size)
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
            {"type": "text", "text": prompt_text}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        processed = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(self.device)
        input_ids = processed["input_ids"]
        attention_mask = processed.get("attention_mask", None)
        pixel_values = processed["pixel_values"]
        image_grid_thw = processed["image_grid_thw"]

        if not target_text:
            return input_ids, None, pixel_values, image_grid_thw, attention_mask

        target_lines = target_text.split("\n")
        ids_list, labels_list = [], []
        for line in target_lines:
            pl = self.processor(text=line + "\n\n", return_tensors="pt")["input_ids"].to(self.device)
            ids_list.append(pl.clone())
            pl[0, -1] = -100  # ignore the final token of each line
            labels_list.append(pl)
        target_labels = torch.cat(labels_list, dim=1)
        labels = torch.cat([torch.full_like(input_ids, -100), target_labels], dim=1)
        target_ids = torch.cat(ids_list, dim=1)
        input_ids = torch.cat([input_ids, target_ids], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(target_ids, device=self.device)], dim=1)

        prompt_len = input_ids.size(1) - target_ids.size(1)
        weights = torch.cat([torch.zeros(prompt_len, device=self.device),
                             torch.linspace(1, 0, steps=target_ids.size(1), device=self.device)], dim=0)
        for i, tid in enumerate(target_ids[0]):
            tok = self.processor.tokenizer.decode([tid.item()]).lower()
            if "".join(c for c in tok if c.isalpha()) in keywords:
                weights[prompt_len + i] += 1.0
        return input_ids, labels, pixel_values, image_grid_thw, attention_mask, weights

    def process_jps_target(self, image_path, prompt_text, target_text):
        """Tokenize JPS's exact user(image+prompt) -> assistant(target) chat.

        Unlike UltraBreak's line-wise target appending, official JPS locates the
        assistant-role boundary in Qwen's full chat template and supervises only
        assistant target tokens before ``<|im_end|>``.
        """
        from qwen_vl_utils import process_vision_info

        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize(self.image_size)
            buffer = io.BytesIO()
            resized.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{encoded}"},
                {"type": "text", "text": prompt_text},
            ],
        }
        user_messages = [user]
        full_messages = [user, {
            "role": "assistant",
            "content": [{"type": "text", "text": target_text}],
        }]
        prefix_text = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(full_messages)
        prefix = self.processor(
            text=[prefix_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)
        full = self.processor(
            text=[full_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        prefix_len = prefix["input_ids"].size(1)
        eos_id = self.processor.tokenizer.eos_token_id
        eos_positions = torch.nonzero(
            full["input_ids"][0, prefix_len:] == eos_id, as_tuple=False
        )
        if eos_positions.numel() == 0:
            raise RuntimeError("JPS target chat has no assistant <|im_end|> token")
        target_stop = prefix_len + int(eos_positions[0].item())
        input_ids = full["input_ids"][:, :target_stop]
        attention_mask = full["attention_mask"][:, :target_stop]
        labels = torch.full_like(input_ids, -100)
        labels[:, prefix_len:target_stop] = input_ids[:, prefix_len:target_stop]
        return (
            input_ids,
            labels,
            full["pixel_values"],
            full["image_grid_thw"],
            attention_mask,
        )

    def preprocess_patched(self, patched_tensor, image_grid_thw, temporal_patch_size=2, patch_size=14, merge_size=2):
        grid_t, grid_h, grid_w = image_grid_thw
        patches = patched_tensor.unsqueeze(0)
        channel = patches.shape[1]
        T = patches.size(0)
        remainder = T % temporal_patch_size
        if remainder != 0:
            pad = temporal_patch_size - remainder
            patches = torch.cat([patches, patches[-1:].expand(pad, *patches.shape[1:])], dim=0)
        patches = patches.reshape(grid_t, temporal_patch_size, channel,
                                  grid_h // merge_size, merge_size, patch_size,
                                  grid_w // merge_size, merge_size, patch_size)
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return patches.reshape(grid_t * grid_h * grid_w,
                               channel * temporal_patch_size * patch_size * patch_size)

    def apply_patch(self, image_tensor_batch, image_grid_thw, patch):
        B, N, D = image_tensor_batch.shape
        while patch.dim() > 3:
            patch = patch.squeeze(0)
        if self.patch_only:
            transformed = self.preprocess_patched(patch, image_grid_thw)
            return transformed.unsqueeze(0).expand(B, -1, -1)
        empty = torch.full((3, self.image_size[0], self.image_size[1]), -1.0,
                           device=patch.device, dtype=patch.dtype, requires_grad=True)
        patched_img = apply_random_patch(empty, patch, scale_range=(0.8, 1.2), rotation_range=(-15, 15))
        transformed = self.preprocess_patched(patched_img, image_grid_thw)
        mask = (transformed != -1.0).float()
        tb = transformed.unsqueeze(0).expand(B, -1, -1)
        mb = mask.unsqueeze(0).expand(B, -1, -1)
        return image_tensor_batch * (1 - mb) + tb * mb

    def compute_loss(self, target, patch, suffix="", print_probs=False):
        input_ids, labels, pixel_values, image_grid_thw, attention_mask, weights = self.process_target(
            target["image"], target["text"] + suffix, target["target"], target["keywords"].split(" "))
        embedding_matrix = self.model.get_input_embeddings().weight
        if self.patch_only:
            mean = torch.tensor(OPENAI_CLIP_MEAN).view(-1, 1, 1).to(self.device)
            std = torch.tensor(OPENAI_CLIP_STD).view(-1, 1, 1).to(self.device)
            patch = (patch - mean) / std
        patched = self.apply_patch(pixel_values.unsqueeze(0), image_grid_thw[0], patch)
        outputs = self.model(input_ids=input_ids, pixel_values=patched,
                             image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        return semantic_similarity_loss(outputs.logits, labels, embedding_matrix,
                                        weights=weights.unsqueeze(0), mode="attention", verbose=print_probs)

    @torch.no_grad()
    def generate(self, prompt, image_path, max_new_tokens=512):
        from qwen_vl_utils import process_vision_info

        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
            {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.device)
        # Some checkpoints persist sampling-only values in generation_config.
        # Explicit greedy defaults silence Transformers' invalid-flag warning
        # without changing deterministic decoding.
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]

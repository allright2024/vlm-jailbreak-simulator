"""Differentiable LLaVA adapter used by JailBound.

The paper evaluates Llama-3.2, Qwen2.5-VL, and MiniGPT-4.  The simulator's
locally pinned white-box target is HF LLaVA-1.5, so this adapter applies the
published algorithm to that model and the manifest records the substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class JointForward:
    features: dict[int, torch.Tensor]
    suffix_embeddings: torch.Tensor
    fluency_loss: torch.Tensor


class LlavaJailBoundAdapter:
    """Expose fusion-layer states and differentiable image/text inputs."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
    ):
        from multimodalmodels.loaders import load_llava

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # All-layer squared-distance losses can overflow FP16 during backward.
        # BF16 preserves the paper's raw-gradient update while retaining the
        # FP32 exponent range; CPU tests use FP32.
        dtype = getattr(torch, torch_dtype) if self.device.type == "cuda" else torch.float32
        self.model, self.processor = load_llava(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map=None,
            to_device=self.device,
            eval_mode=True,
        )
        self.model.requires_grad_(False)
        language_model = self.model.model.language_model
        layers = language_model.layers if hasattr(language_model, "layers") else language_model.model.layers
        self.n_layers = len(layers)
        self.embedding = self.model.get_input_embeddings()
        self._dtype = next(self.model.parameters()).dtype
        special = set(getattr(self.processor.tokenizer, "all_special_ids", []))
        special.add(int(self.model.config.image_token_id))
        self._forbidden_token_ids = sorted(i for i in special if i >= 0)

    def _prompt_parts(self, instruction: str) -> tuple[str, str]:
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        marker = "ASSISTANT:"
        split = prompt.rfind(marker)
        if split < 0:
            raise ValueError("LLaVA chat template does not contain the ASSISTANT marker")
        return prompt[:split], prompt[split:]

    def prepare_context(
        self, image: Image.Image, instruction: str, suffix_ids: torch.Tensor | None = None
    ) -> dict[str, Any]:
        prefix, postfix = self._prompt_parts(instruction)
        prefix_inputs = self.processor(images=image, text=prefix, return_tensors="pt")
        prefix_ids = prefix_inputs["input_ids"]
        postfix_ids = self.processor.tokenizer(
            postfix, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        pieces = [prefix_ids]
        suffix_start = int(prefix_ids.shape[1])
        suffix_end = suffix_start
        if suffix_ids is not None:
            suffix_cpu = suffix_ids.detach().cpu().long().view(1, -1)
            pieces.append(suffix_cpu)
            suffix_end += int(suffix_cpu.shape[1])
        pieces.append(postfix_ids)
        ids = torch.cat(pieces, dim=1).to(self.device)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "suffix_start": suffix_start,
            "suffix_end": suffix_end,
        }

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        pixel_values = self.processor.image_processor(
            image.convert("RGB"), return_tensors="pt"
        )["pixel_values"].to(self.device)
        mean = torch.tensor(
            self.processor.image_processor.image_mean,
            device=self.device,
            dtype=pixel_values.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.processor.image_processor.image_std,
            device=self.device,
            dtype=pixel_values.dtype,
        ).view(1, 3, 1, 1)
        return (pixel_values * std + mean).clamp(0, 1).float()

    def _normalize(self, image01: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            self.processor.image_processor.image_mean,
            device=image01.device,
            dtype=image01.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.processor.image_processor.image_std,
            device=image01.device,
            dtype=image01.dtype,
        ).view(1, 3, 1, 1)
        return ((image01 - mean) / std).to(self._dtype)

    @staticmethod
    def _pool_hidden(
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        layer_indexes: list[int],
    ) -> dict[int, torch.Tensor]:
        # The paper does not state how the sequence dimension of phi^(l) is
        # reduced.  The final non-padding input token is the causal state that
        # has attended to the complete image+text pair, so the clean-room port
        # uses that explicitly and records it in effective_config.
        position = int(attention_mask[0].sum().item()) - 1
        return {
            layer: hidden_states[layer + 1][0, position].float()
            for layer in layer_indexes
        }

    def extract_features(
        self,
        image01: torch.Tensor,
        context: dict[str, Any],
        layer_indexes: list[int],
    ) -> dict[int, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(
                input_ids=context["input_ids"],
                attention_mask=context["attention_mask"],
                pixel_values=self._normalize(image01),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return {
            key: value.detach()
            for key, value in self._pool_hidden(
                outputs.hidden_states, context["attention_mask"], layer_indexes
            ).items()
        }

    def forward_joint(
        self,
        image01: torch.Tensor,
        context: dict[str, Any],
        suffix_ids: torch.Tensor,
        layer_indexes: list[int],
    ) -> JointForward:
        suffix_ids = suffix_ids.to(self.device).long().view(-1)
        start, end = context["suffix_start"], context["suffix_end"]
        if end - start != suffix_ids.numel():
            raise ValueError("JailBound suffix length changed after context construction")
        ids = context["input_ids"].clone()
        ids[0, start:end] = suffix_ids
        fixed = self.embedding(ids).detach()
        suffix_embeddings = self.embedding(suffix_ids).detach().clone().requires_grad_(True)
        inputs_embeds = torch.cat(
            [fixed[:, :start], suffix_embeddings.unsqueeze(0), fixed[:, end:]], dim=1
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=context["attention_mask"],
            pixel_values=self._normalize(image01),
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        features = self._pool_hidden(
            outputs.hidden_states, context["attention_mask"], layer_indexes
        )
        # L_suffix is left unspecified in Eq. (8).  We instantiate it as the
        # conditional next-token NLL over suffix positions, a standard fluency
        # regularizer, and expose that choice in the manifest.
        predictors = outputs.logits[0, start - 1:end - 1].float()
        fluency = F.cross_entropy(predictors, suffix_ids, reduction="mean")
        return JointForward(features, suffix_embeddings, fluency)

    def random_suffix(self, length: int, generator: torch.Generator) -> torch.Tensor:
        vocab = int(self.embedding.weight.shape[0])
        ids = torch.randint(0, vocab, (length,), generator=generator, device=self.device)
        forbidden = set(self._forbidden_token_ids)
        # Deterministic rejection keeps all initialized tokens in the ordinary
        # vocabulary while preserving the paper's random initialization.
        for position in range(length):
            while int(ids[position]) in forbidden:
                ids[position] = torch.randint(
                    0, vocab, (1,), generator=generator, device=self.device
                )[0]
        return ids

    @torch.no_grad()
    def nearest_tokens(self, desired: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Eq. (11), exact full-vocabulary nearest-neighbour in chunks."""
        desired = desired.detach().to(self.device).float()
        weights = self.embedding.weight.detach()
        best_distance = torch.full(
            (desired.shape[0],), float("inf"), device=self.device
        )
        best_ids = torch.zeros(desired.shape[0], dtype=torch.long, device=self.device)
        forbidden = set(self._forbidden_token_ids)
        for start in range(0, weights.shape[0], chunk_size):
            stop = min(start + chunk_size, weights.shape[0])
            chunk = weights[start:stop].float()
            distances = (
                chunk.square().sum(dim=1, keepdim=True)
                - 2.0 * chunk @ desired.T
            )
            for token_id in forbidden:
                if start <= token_id < stop:
                    distances[token_id - start] = float("inf")
            values, offsets = distances.min(dim=0)
            improved = values < best_distance
            best_distance[improved] = values[improved]
            best_ids[improved] = offsets[improved] + start
        return best_ids

    def decode_suffix(self, suffix_ids: torch.Tensor) -> str:
        return self.processor.tokenizer.decode(
            suffix_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

"""Targets for native Hugging Face multimodal chat checkpoints.

The three families here expose the same :class:`core.model.TargetModel`
surface while retaining their official processor chat templates.  Model
construction is centralized in :mod:`multimodalmodels.loaders` and therefore
uses the pinned repository revisions from ``multimodalmodels/registry.py``.
"""

from __future__ import annotations

from typing import Sequence

from ..model import TargetModel
from ..paths import resolve_path


class NativeHFVLMTarget(TargetModel):
    """Shared deterministic generation path for modern native-HF VLMs."""

    model_id: str = ""
    revision: str | None = None
    max_images: int | None = None

    def __init__(self, max_new_tokens: int = 256):
        from multimodalmodels.loaders import load_native_vlm

        self.model, self.processor = load_native_vlm(
            self.model_id,
            revision=self.revision,
            torch_dtype="auto",
            device_map="auto",
            eval_mode=True,
        )
        self.max_new_tokens = max_new_tokens
        self.do_sample = False
        self.temperature = 1.0
        self.top_p = 1.0

    def _images(self, image_paths: Sequence[str]):
        from PIL import Image

        if self.max_images is not None and len(image_paths) > self.max_images:
            raise ValueError(
                f"{self.name} accepts at most {self.max_images} image(s); "
                f"got {len(image_paths)}"
            )
        # Copy pixels before closing the files.  Processor chat templates can
        # then consume ordinary PIL objects without holding file descriptors.
        images = []
        for path in image_paths:
            with Image.open(resolve_path(path)) as image:
                images.append(image.convert("RGB").copy())
        return images

    @staticmethod
    def _messages(text: str, images) -> list[dict]:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def generate(self, text: str, image_paths: Sequence[str]) -> str:
        import torch

        messages = self._messages(text, self._images(image_paths))
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generation = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation["temperature"] = self.temperature
            if self.top_p is not None:
                generation["top_p"] = self.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generate_batch(
        self, prompts: Sequence[tuple[str, Sequence[str]]]
    ) -> list[str]:
        # Sequential execution keeps heterogeneous aspect ratios and image
        # counts valid across all three processor families.
        return [self.generate(text, images) for text, images in prompts]


class LlavaNextTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import LLAVA_16_HF_ID, LLAVA_16_REVISION

    name = "llava_v1_6"
    model_id = LLAVA_16_HF_ID
    revision = LLAVA_16_REVISION
    max_images = 1


class Qwen3VL2BTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import QWEN3_VL_2B_HF_ID, QWEN3_VL_2B_REVISION

    name = "qwen3_vl_2b"
    model_id = QWEN3_VL_2B_HF_ID
    revision = QWEN3_VL_2B_REVISION


class Qwen3VL8BTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import QWEN3_VL_8B_HF_ID, QWEN3_VL_8B_REVISION

    name = "qwen3_vl_8b"
    model_id = QWEN3_VL_8B_HF_ID
    revision = QWEN3_VL_8B_REVISION


class InternVL3_2BTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import INTERNVL3_2B_HF_ID, INTERNVL3_2B_REVISION

    name = "internvl3_2b"
    model_id = INTERNVL3_2B_HF_ID
    revision = INTERNVL3_2B_REVISION


class InternVL3_8BTarget(NativeHFVLMTarget):
    from multimodalmodels.registry import INTERNVL3_8B_HF_ID, INTERNVL3_8B_REVISION

    name = "internvl3_8b"
    model_id = INTERNVL3_8B_HF_ID
    revision = INTERNVL3_8B_REVISION


__all__ = [
    "NativeHFVLMTarget",
    "LlavaNextTarget",
    "Qwen3VL2BTarget",
    "Qwen3VL8BTarget",
    "InternVL3_2BTarget",
    "InternVL3_8BTarget",
]

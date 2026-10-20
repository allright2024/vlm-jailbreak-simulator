"""Target (victim) VLM wrappers exposing the core.TargetModel interface.

These adapt the existing wrappers under multimodalmodels/ so the pipeline can
call ``generate(text, image_paths)`` with repository-relative or absolute paths.
Importing this package registers the wrappers in TARGET_REGISTRY.
"""

from .llava import LlavaTarget  # noqa: F401
from .native_hf import (InternVL3_2BTarget, InternVL3_8BTarget, LlavaNextTarget,
                        Qwen3VL2BTarget, Qwen3VL8BTarget)  # noqa: F401
from .openai_vision import OpenAIVisionTarget  # noqa: F401
from .qwen2vl import Qwen2VLTarget, Qwen25VLTarget  # noqa: F401

from multimodalmodels.registry import (INTERNVL3_2B_HF_ID, INTERNVL3_8B_HF_ID,
                                       LLAVA_15_HF_ID, LLAVA_16_HF_ID,
                                       QWEN2_VL_HF_ID, QWEN25_VL_HF_ID,
                                       QWEN3_VL_2B_HF_ID, QWEN3_VL_8B_HF_ID)

TARGET_REGISTRY = {
    "llava_v1_5": LlavaTarget,
    "LLaVA_v1_5": LlavaTarget,
    LLAVA_15_HF_ID: LlavaTarget,
    "llava_v1_6": LlavaNextTarget,
    "LLaVA_v1_6": LlavaNextTarget,
    "llava_next_7b": LlavaNextTarget,
    LLAVA_16_HF_ID: LlavaNextTarget,
    "qwen2_vl_7b": Qwen2VLTarget,
    "Qwen2-VL-7B-Instruct": Qwen2VLTarget,
    QWEN2_VL_HF_ID: Qwen2VLTarget,
    "qwen2_5_vl_7b": Qwen25VLTarget,
    "Qwen2.5-VL-7B-Instruct": Qwen25VLTarget,
    QWEN25_VL_HF_ID: Qwen25VLTarget,
    "qwen3_vl_2b": Qwen3VL2BTarget,
    "Qwen3-VL-2B-Instruct": Qwen3VL2BTarget,
    QWEN3_VL_2B_HF_ID: Qwen3VL2BTarget,
    "qwen3_vl_8b": Qwen3VL8BTarget,
    "Qwen3-VL-8B-Instruct": Qwen3VL8BTarget,
    QWEN3_VL_8B_HF_ID: Qwen3VL8BTarget,
    "internvl3_2b": InternVL3_2BTarget,
    INTERNVL3_2B_HF_ID: InternVL3_2BTarget,
    "internvl3_8b": InternVL3_8BTarget,
    INTERNVL3_8B_HF_ID: InternVL3_8BTarget,
    "gpt-4o-mini": lambda: OpenAIVisionTarget("gpt-4o-mini"),
    "gpt-4o": lambda: OpenAIVisionTarget("gpt-4o"),
    "gpt-4o-2024-08-06": lambda: OpenAIVisionTarget("gpt-4o-2024-08-06"),
    "gpt-4-turbo-2024-04-09": lambda: OpenAIVisionTarget("gpt-4-turbo-2024-04-09"),
}


def load_target(name: str):
    if name not in TARGET_REGISTRY:
        raise KeyError(f"Unknown target model {name!r}. Available: {sorted(TARGET_REGISTRY)}")
    return TARGET_REGISTRY[name]()


__all__ = [
    "LlavaTarget", "LlavaNextTarget", "OpenAIVisionTarget", "Qwen2VLTarget",
    "Qwen25VLTarget", "Qwen3VL2BTarget", "Qwen3VL8BTarget",
    "InternVL3_2BTarget", "InternVL3_8BTarget", "TARGET_REGISTRY", "load_target",
]

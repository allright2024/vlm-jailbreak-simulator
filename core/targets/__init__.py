"""Target (victim) VLM wrappers exposing the core.TargetModel interface.

These adapt the existing wrappers under multimodalmodels/ so the pipeline can
call ``generate(text, image_paths)`` with repository-relative or absolute paths.
Importing this package registers the wrappers in TARGET_REGISTRY.
"""

from .llava import LlavaTarget  # noqa: F401

TARGET_REGISTRY = {
    "llava_v1_5": LlavaTarget,
    "LLaVA_v1_5": LlavaTarget,
}


def load_target(name: str):
    if name not in TARGET_REGISTRY:
        raise KeyError(f"Unknown target model {name!r}. Available: {sorted(TARGET_REGISTRY)}")
    return TARGET_REGISTRY[name]()


__all__ = ["LlavaTarget", "TARGET_REGISTRY", "load_target"]

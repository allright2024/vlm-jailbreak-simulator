"""VLMShield input-filter adapter for the unified pipeline.

VLMShield (Qi et al., arXiv:2604.06502; pgqihere/VLMShield@344ca14) classifies
a (text, image) pair as safe / unsafe from MAFE features: CLIP ViT-L/14 image
[CLS] embedding (768) + chunked, similarity-weighted text [EOS] embedding (768)
-> released 3-layer MLP.  It never queries the target model, so a pair that is
predicted unsafe is blocked with a fixed refusal and a passing pair reaches the
target unchanged.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .base import BaseDefense
from .vlmshield_utils import (
    CLIP_EMBED_DIM,
    EMBEDDING_DIM,
    ThreeLayerClassifier,
    classify,
    multimodal_embedding,
)


OFFICIAL_REPO = "https://github.com/pgqihere/VLMShield"
OFFICIAL_COMMIT = "344ca14778383aef57315229118f7669454fb4bb"
OFFICIAL_CLIP_MODEL = "openai/clip-vit-large-patch14"
OFFICIAL_CLASSIFIER_NAME = "vlmshield_classifier.pt"
OFFICIAL_CLASSIFIER_SHA256 = (
    "b36702877791f347f0b846d93b9b9162c86a8895efe962b6052d322ed0c53c34"
)
DEFAULT_CLASSIFIER_PATH = (
    Path(__file__).resolve().parent / "vlmshield_assets" / OFFICIAL_CLASSIFIER_NAME
)
# Upstream only computes detection rates; the refusal text that a blocked pair
# receives in the simulator is the same fixed sentence the JailGuard adapter uses.
DEFAULT_REFUSAL = "I'm sorry, but I cannot assist with that request."
LABEL_MEANING = {0: "unsafe", 1: "safe"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VLMShieldDefense(BaseDefense):
    """Official-profile VLMShield detector (CLIP ViT-L/14 + released MLP)."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._clip = None
        self._processor = None
        self._classifier = None
        self._classifier_digest: str | None = None
        self._device = None

    # ------------------------------------------------------------ resources
    def _resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / path
        if candidate.exists():
            return candidate.resolve()
        config_path = self.config.get("_config_path")
        if config_path:
            config_candidate = Path(config_path).resolve().parent / path
            if config_candidate.exists():
                return config_candidate.resolve()
        return candidate.resolve()

    def _get_device(self):
        if self._device is None:
            import torch

            configured = self.config.get("device")
            if configured is None:
                configured = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._device = torch.device(str(configured))
        return self._device

    def _get_clip(self):
        if self._clip is None:
            from multimodalmodels.loaders import load_clip_model

            model_id = str(self.config.get("clip_model", OFFICIAL_CLIP_MODEL))
            if model_id != OFFICIAL_CLIP_MODEL:
                raise ValueError(
                    f"Official VLMShield uses {OFFICIAL_CLIP_MODEL}; got {model_id!r}."
                )
            self._clip, self._processor = load_clip_model(
                model_id,
                revision=self.config.get("clip_revision"),
                device=self._get_device(),
            )
        return self._clip, self._processor

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier
        import torch

        configured = self.config.get("classifier_path", str(DEFAULT_CLASSIFIER_PATH))
        path = self._resolve_path(str(configured))
        if not path.is_file():
            raise FileNotFoundError(
                f"VLMShield classifier not found: {path}. "
                "Run: bash defense_scripts/setup_vlmshield.sh"
            )
        digest = _sha256_file(path)
        expected = str(
            self.config.get("expected_classifier_sha256", OFFICIAL_CLASSIFIER_SHA256)
        )
        if expected and digest != expected:
            raise ValueError(
                "VLMShield classifier SHA-256 mismatch: "
                f"expected {expected}, got {digest} ({path})."
            )
        classifier = ThreeLayerClassifier(dim=EMBEDDING_DIM)
        state = torch.load(path, map_location="cpu", weights_only=True)
        classifier.load_state_dict(state)
        self._classifier = classifier.to(self._get_device()).eval()
        self._classifier_digest = digest
        return self._classifier

    # ------------------------------------------------------------ detection
    def _predict(self, text: str, image_path: str | None):
        """Return (prediction, confidence, probs, has_image, has_text, embedding)."""
        from PIL import Image

        clip_model, processor = self._get_clip()
        classifier = self._get_classifier()
        image = None
        if image_path:
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"VLMShield input image not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
        embedding, has_image, has_text = multimodal_embedding(
            clip_model, processor, text, image, self._get_device()
        )
        prediction, confidence, probs = classify(classifier, embedding)
        return prediction, confidence, probs, has_image, has_text, embedding

    def _apply_pair(self, text: str, image_path: str | None) -> dict[str, Any]:
        prediction, confidence, probs, has_image, has_text, embedding = self._predict(
            text, image_path
        )
        blocked = prediction == 0
        refusal = str(self.config.get("refusal_response", DEFAULT_REFUSAL))
        meta: dict[str, Any] = {
            "defense": "VLMShield",
            "detected_adversarial": blocked,
            "prediction": prediction,
            "prediction_label": LABEL_MEANING[prediction],
            "confidence": confidence,
            "probabilities": {"unsafe": probs[0], "safe": probs[1]},
            "has_image": has_image,
            "has_text": has_text,
            "clip_model": OFFICIAL_CLIP_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "classifier_sha256": self._classifier_digest,
            "query_budget": 0,
            "official_repo": OFFICIAL_REPO,
            "official_commit": OFFICIAL_COMMIT,
            "algorithm_fidelity": "faithful_port",
        }
        if self.config.get("store_embedding", False):
            meta["embedding"] = [float(v) for v in embedding.detach().cpu().tolist()]
        return {
            "blocked": blocked,
            "text": text,
            "images": [image_path] if image_path else [],
            "default_response": refusal if blocked else None,
            "meta": meta,
        }

    # ------------------------------------------------------------ interfaces
    def apply(self, image_filename, prompt, model=None, image_dir=None, **kwargs):
        image_path = None
        if image_filename:
            image_path = (
                image_filename
                if os.path.isabs(image_filename) or image_dir is None
                else os.path.join(image_dir, image_filename)
            )
        result = self._apply_pair(prompt, image_path)
        return {
            "blocked": result["blocked"],
            "image_filename": image_filename,
            "prompt": result["text"],
            "default_response": result["default_response"],
            "meta": result["meta"],
        }

    def apply_core(self, test_case, target=None) -> dict[str, Any]:
        if len(test_case.images) > 1:
            raise ValueError(
                "VLMShield's adapter scores one image per query (zero images is "
                f"the text-only path); got {len(test_case.images)}."
            )
        image_path = None
        if test_case.images:
            from core.paths import resolve_path

            image_path = resolve_path(test_case.images[0])
        result = self._apply_pair(test_case.text, image_path)
        # A passing pair reaches the downstream target unchanged.
        result["images"] = list(test_case.images)
        return result


__all__ = [
    "VLMShieldDefense",
    "OFFICIAL_REPO",
    "OFFICIAL_COMMIT",
    "OFFICIAL_CLIP_MODEL",
    "OFFICIAL_CLASSIFIER_SHA256",
    "DEFAULT_REFUSAL",
    "CLIP_EMBED_DIM",
]

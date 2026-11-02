"""Paper-style CLIP embedding matching, separate from the unchanged released demo.

All four trigger strategies have image outputs. Text targets use CLIP's text
encoder; OCR, visual and combined targets use its image encoder. Adam updates
the normalized model input described in Appendix H.3. Projection bounds map
to RGB pixels in [0, 1]. Exact historical input handling is unknown.

The caller supplies the target data. The included experiment runner uses only
benign geometric controls; this module does not supply the paper's dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from multimodalmodels import loaders
from multimodalmodels.registry import CLIP_VIT_L14_336_HF_ID, CLIP_VIT_L14_336_REVISION

from .jailbreak_in_pieces import embedding_distance

TRIGGER_TYPES = ("textual", "ocr", "visual", "combined")


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class EmbeddingTarget:
    kind: str
    text: str | None = None
    image: str | None = None

    def __post_init__(self):
        if self.kind not in TRIGGER_TYPES:
            raise ValueError(f"unknown trigger type: {self.kind}")
        if self.kind == "textual":
            if not self.text or self.image is not None:
                raise ValueError("textual target requires text and no image")
        elif not self.image or self.text is not None:
            raise ValueError("image target requires an image and no text encoder input")


@dataclass(frozen=True)
class MatchConfig:
    max_steps: int = 500
    threshold: float = 0.3
    learning_rate: float = 0.1
    normalize: bool = True
    seed: int = 2024

    def __post_init__(self):
        if self.max_steps < 1 or self.threshold < 0 or self.learning_rate <= 0:
            raise ValueError("invalid optimization bounds")


def normalize_pixels(pixels: torch.Tensor, mean, std) -> torch.Tensor:
    mean = pixels.new_tensor(mean).view(1, 3, 1, 1)
    std = pixels.new_tensor(std).view(1, 3, 1, 1)
    return (pixels - mean) / std


class PaperEmbeddingMatcher:
    """Raw projected CLIP L2 objective with explicit best-state and file losses."""

    def __init__(self, config: MatchConfig, device="cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.config = config
        self.device = torch.device(device)
        self.model, self.processor = loaders.load_clip_image_encoder(
            CLIP_VIT_L14_336_HF_ID, revision=CLIP_VIT_L14_336_REVISION,
            device=self.device, do_normalize=config.normalize, local_files_only=True,
        )
        self.tokenizer = loaders.load_clip_tokenizer()
        self.model.requires_grad_(False)

    def raw_pixels(self, path: str | Path) -> torch.Tensor:
        with Image.open(path) as image:
            batch = self.processor(images=image.convert("RGB"), do_normalize=False, return_tensors="pt")
        return batch["pixel_values"].to(self.device, dtype=torch.float32)

    def processed(self, pixels):
        if not self.config.normalize:
            return pixels
        return normalize_pixels(pixels, self.processor.image_mean, self.processor.image_std)

    def image_embedding(self, pixels):
        return self.model.get_image_features(self.processed(pixels))

    def rgb_pixels(self, model_input):
        if not self.config.normalize:
            return model_input
        mean = model_input.new_tensor(self.processor.image_mean).view(1, 3, 1, 1)
        std = model_input.new_tensor(self.processor.image_std).view(1, 3, 1, 1)
        return model_input * std + mean

    @torch.no_grad()
    def target_embedding(self, target: EmbeddingTarget):
        if target.kind == "textual":
            # Reject silently truncated descriptions: they change the target.
            tokens = self.tokenizer(target.text, truncation=False, return_tensors="pt")
            if tokens["input_ids"].shape[1] > self.model.config.text_config.max_position_embeddings:
                raise ValueError("text target exceeds the CLIP context length")
            return self.model.get_text_features(**tokens.to(self.device))
        return self.image_embedding(self.raw_pixels(target.image))

    def optimize(self, target: EmbeddingTarget, base_image: str | Path, out_dir: str | Path) -> dict:
        folder = Path(out_dir)
        folder.mkdir(parents=True, exist_ok=True)
        signature = {"config": asdict(self.config), "target": asdict(target),
                     "target_image_sha256": file_digest(target.image) if target.image else None,
                     "base_sha256": file_digest(base_image), "encoder": CLIP_VIT_L14_336_HF_ID,
                     "encoder_revision": CLIP_VIT_L14_336_REVISION,
                     "implementation_sha256": file_digest(__file__), "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else str(self.device),
                     "optimization_space": "CLIP model input; normalized when normalize=True",
                     "projection": "bounds corresponding to valid RGB pixels"}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        manifest = folder / "artifact.json"
        if manifest.exists():
            cached = json.loads(manifest.read_text())
            if cached["signature_sha256"] != key:
                raise ValueError(f"cached optimization settings changed: {folder}")
            for spec in cached["files"].values():
                if file_digest(spec["path"]) != spec["sha256"]:
                    raise ValueError(f"cached artifact changed: {spec['path']}")
            return cached

        torch.manual_seed(self.config.seed)
        target_emb = self.target_embedding(target).detach()
        pixels = self.raw_pixels(base_image)
        variable = self.processed(pixels).detach().clone().requires_grad_(True)
        lower, upper = self.processed(torch.zeros_like(pixels)), self.processed(torch.ones_like(pixels))
        optimizer = torch.optim.Adam([variable], lr=self.config.learning_rate)
        best_pixels, best_loss = pixels.detach().clone(), float("inf")
        history, updates = [], 0

        def observe(step, value):
            nonlocal best_pixels, best_loss
            score = float(value.detach().cpu())
            if not torch.isfinite(value):
                raise RuntimeError("non-finite embedding loss")
            history.append({"updates": step, "loss": score})
            if score < best_loss:
                best_loss, best_pixels = score, self.rgb_pixels(variable).detach().clamp(0, 1).clone()
            return score

        for step in range(self.config.max_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = embedding_distance(self.model.get_image_features(variable), target_emb).mean()
            score = observe(step, loss)
            if step % 100 == 0:
                print(f"{target.kind}: update={step} loss={score:.5f} best={best_loss:.5f}", flush=True)
            if score <= self.config.threshold:
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                variable.copy_(torch.maximum(lower, torch.minimum(upper, variable)))
            updates += 1
        with torch.no_grad():
            observe(updates, embedding_distance(self.model.get_image_features(variable), target_emb).mean())
            transported_loss = float(embedding_distance(self.image_embedding(best_pixels), target_emb).mean())

        tensor_path = folder / "pixels.safetensors"
        save_file({"pixels": best_pixels.detach().cpu().contiguous()}, str(tensor_path),
                  metadata={"space": "RGB float32 in [0,1] before CLIP normalization"})
        image = Image.fromarray(np.asarray(
            (best_pixels[0].detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        ))
        png_path, jpeg_path = folder / "preview.png", folder / "preview.jpg"
        image.save(png_path)
        image.save(jpeg_path)
        with torch.no_grad():
            file_losses = {name: float(embedding_distance(self.image_embedding(self.raw_pixels(path)), target_emb).mean())
                           for name, path in (("png_l2", png_path), ("jpeg_l2", jpeg_path))}
        history_path = folder / "loss_history.json"
        history_path.write_text(json.dumps(history) + "\n")
        artifact = {
            "signature": signature, "signature_sha256": key, "trigger_type": target.kind,
            "initial_l2": history[0]["loss"], "best_tensor_l2": best_loss,
            "float_transport_l2": transported_loss,
            "last_tensor_l2": history[-1]["loss"], "updates": updates,
            "converged": best_loss <= self.config.threshold, **file_losses,
            "checkpoint_rule": "minimum evaluated L2; cap is an explicit control-experiment choice",
            "files": {name: {"path": str(path.resolve()), "sha256": file_digest(path)}
                      for name, path in (("tensor", tensor_path), ("png", png_path),
                                         ("jpeg", jpeg_path), ("history", history_path))},
        }
        manifest.write_text(json.dumps(artifact, indent=2) + "\n")
        return artifact

"""Unified CIDER detector adapter.

Detection follows PandragonXIII/CIDER: compare harmful-query/image cosine
similarity at diffusion checkpoints 0, 50, ..., 350 and reject whenever a
drop is below the calibrated threshold.
"""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import BaseDefense
from .cider_utils import (
    Defender,
    Encoder,
    generate_denoised_img_diffusion,
    get_denoiser,
)


OFFICIAL_REPO = "https://github.com/PandragonXIII/CIDER"
OFFICIAL_COMMIT = "3c08d7e2d52e27b47d1e428a030e8ffda5efae3e"
OFFICIAL_REFUSAL = "I'm sorry, but I can not assist."
DEFAULT_DENOISER_PATH = (
    Path(__file__).resolve().parent
    / "cider_models/diffusion_denoiser/imagenet/256x256_diffusion_uncond.pt"
)
OFFICIAL_DENOISER_SIZE = 2_211_383_297
OFFICIAL_DENOISER_MD5 = "fd9dd2335b8736d521de0aed54bd90ca"


class ReusedLlavaEncoder(Encoder):
    def __init__(self, llava_model):
        backend = getattr(llava_model, "_model", llava_model)
        if not hasattr(backend, "model") or not hasattr(backend, "processor"):
            raise TypeError(
                "CIDER currently requires the registered LLaVA-1.5 target; "
                "the target wrapper does not expose its model and processor."
            )
        self.m = backend.model
        self.processor = backend.processor
        self.device = self.m.device

    def embed_img(self, imgpth) -> np.ndarray:
        image = Image.open(imgpth).convert("RGB")
        pixel_value = self.processor.image_processor(
            image, return_tensors="pt"
        ).pixel_values.to(self.device, dtype=self.m.dtype)
        with torch.no_grad():
            image_outputs = self.m.vision_tower(
                pixel_value, output_hidden_states=True
            )
            selected = image_outputs.hidden_states[
                self.m.config.vision_feature_layer
            ]
            selected = selected[:, 1:]
            feats = self.m.multi_modal_projector(selected)
            feats = torch.mean(feats, dim=1).to("cpu").float().numpy()
        return feats

    def embed_text(self, text) -> np.ndarray:
        input_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(
            self.device
        )
        with torch.no_grad():
            embeds = self.m.get_input_embeddings()(input_ids)
            embeds = torch.mean(embeds, dim=1).to("cpu").float().numpy()
        return embeds

    def calc_cossim(self, pairs):
        res = []
        for text, imgpth in pairs:
            t = self.embed_text(text)
            im = self.embed_img(imgpth)
            res.append(self.compute_cosine(t, im))
        return res


class CIDERDefense(BaseDefense):
    def __init__(self, config=None):
        super().__init__(config)
        self._encoder = None
        self._denoiser = None

    def _device(self) -> str:
        value = self.config.get("cuda", 0)
        if isinstance(value, int):
            return f"cuda:{value}"
        return str(value)

    def _denoiser_path(self) -> Path:
        configured = self.config.get("denoiser_path")
        if not configured:
            return DEFAULT_DENOISER_PATH
        path = Path(os.path.expandvars(os.path.expanduser(str(configured))))
        if path.is_absolute():
            return path
        repo_candidate = Path(__file__).resolve().parent.parent / path
        if repo_candidate.exists():
            return repo_candidate.resolve()
        config_path = self.config.get("_config_path")
        if config_path:
            config_candidate = Path(config_path).resolve().parent / path
            if config_candidate.exists():
                return config_candidate.resolve()
        return repo_candidate.resolve()

    def _get_encoder(self, model):
        if self._encoder is None:
            self._encoder = ReusedLlavaEncoder(model)
        return self._encoder

    def _get_denoiser(self):
        if self._denoiser is not None:
            return self._denoiser
        checkpoint = self._denoiser_path()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "CIDER diffusion checkpoint is missing: "
                f"{checkpoint}. Run: bash defense_scripts/setup_cider.sh"
            )
        expected_size = int(
            self.config.get("expected_denoiser_size", OFFICIAL_DENOISER_SIZE)
        )
        if expected_size and checkpoint.stat().st_size != expected_size:
            raise ValueError(
                f"CIDER checkpoint size mismatch: expected {expected_size}, "
                f"got {checkpoint.stat().st_size} ({checkpoint})"
            )
        if self.config.get("verify_denoiser_md5", False):
            hasher = hashlib.md5()
            with checkpoint.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest != OFFICIAL_DENOISER_MD5:
                raise ValueError(
                    "CIDER checkpoint MD5 mismatch: "
                    f"expected {OFFICIAL_DENOISER_MD5}, got {digest}"
                )
        self._denoiser = get_denoiser(
            device=self._device(), model_path=str(checkpoint)
        )
        return self._denoiser

    def _apply_path(self, image_path: str, prompt: str, model) -> dict:
        threshold = float(
            self.config.get("default_threshold", -0.003936767578125)
        )
        cpnum = int(self.config.get("cpnum", 8))
        step = int(self.config.get("step", 50))
        if cpnum != 8 or step != 50:
            raise ValueError(
                "Official CIDER uses cpnum=8 and step=50 "
                "(checkpoints 0, 50, ..., 350)."
            )
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"CIDER input image not found: {image_path}")

        encoder = self._get_encoder(model)
        denoiser = self._get_denoiser()
        tmp_parent = self.config.get("temp_dir")
        tmp_dir = tempfile.mkdtemp(prefix="cider_", dir=tmp_parent)
        try:
            denoised_dir = os.path.join(tmp_dir, "denoised_imgs")
            os.makedirs(denoised_dir, exist_ok=True)
            denoised_pths = generate_denoised_img_diffusion(
                imgpth=[image_path],
                save_dir=denoised_dir,
                cps=cpnum,
                step=step,
                device=self._device(),
                batch_size=50,
                denoiser=denoiser,
            )
            denoised_pths.sort()
            cossims = [float(v) for v in encoder.calc_cossim(
                [(prompt, path) for path in denoised_pths]
            )]
            if len(cossims) != cpnum:
                raise RuntimeError(
                    f"CIDER expected {cpnum} similarity checkpoints, "
                    f"got {len(cossims)}."
                )
            detector = Defender(threshold=threshold)
            low_idx = int(detector.get_lowest_idx(cossims))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        deltas = [value - cossims[0] for value in cossims]
        is_adversarial = low_idx != 0
        return {
            "blocked": is_adversarial,
            "text": prompt,
            "images": [image_path],
            "default_response": OFFICIAL_REFUSAL if is_adversarial else None,
            "meta": {
                "defense": "CIDER",
                "detected_adversarial": is_adversarial,
                "threshold": threshold,
                "cpnum": cpnum,
                "step": step,
                "denoise_checkpoints": [i * step for i in range(cpnum)],
                "cosine_similarities": cossims,
                "cosine_deltas": deltas,
                "lowest_index": low_idx,
                "official_repo": OFFICIAL_REPO,
                "official_commit": OFFICIAL_COMMIT,
                "algorithm_fidelity": "faithful_port",
            },
        }

    def apply(self, image_filename, prompt, model=None, image_dir=None, **kwargs):
        if model is None or image_dir is None:
            raise ValueError(
                "CIDERDefense.apply() requires `model`(LLaVA) and `image_dir` "
                "(denoise된 이미지를 인코딩하는 데 필요)."
            )

        src_image_path = (
            image_filename
            if os.path.isabs(image_filename)
            else os.path.join(image_dir, image_filename)
        )
        result = self._apply_path(src_image_path, prompt, model)
        return {
            "blocked": result["blocked"],
            "image_filename": image_filename,
            "prompt": result["text"],
            "default_response": result["default_response"],
            "meta": result["meta"],
        }

    def apply_core(self, test_case, target=None) -> dict:
        if target is None:
            raise ValueError("CIDER.apply_core() requires the loaded LLaVA target.")
        if len(test_case.images) != 1:
            raise ValueError(
                f"CIDER's injection-mode adapter requires exactly one image; "
                f"got {len(test_case.images)}."
            )
        from core.paths import resolve_path

        image_path = resolve_path(test_case.images[0])
        result = self._apply_path(image_path, test_case.text, target)
        # The paper specifies that a passing pair reaches the MLLM unchanged.
        result["images"] = list(test_case.images)
        return result

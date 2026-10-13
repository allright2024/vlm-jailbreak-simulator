"""Visual Adversarial Examples (VisualAdv, Qi et al., AAAI 2024) on the core interface.

The foundational white-box image jailbreak: PGD-optimize a perturbation on a
benign image so the VLM is driven to produce toxic content from a small
derogatory corpus (cross-entropy on those target strings). The resulting single
universal adversarial image, paired with any harmful instruction at eval time,
tends to make the model comply.

Ported from the paper's method and the repo's existing image_attacks/visual_adv
(constrained PGD: L_inf eps budget, keep the image in [0,1]). Reuses the LLaVA
wrapper's compute_loss_batch as the white-box surrogate loss.

The image is optimized once (over the corpus) and cached, then reused for every
behavior. requires_surrogate; white-box when the eval target is the surrogate,
transfer otherwise.
"""

from __future__ import annotations

import csv
import os
import random

import torch
from PIL import Image
from torchvision.utils import save_image

from core.attack import Attack
from core.dataset import DATA_ROOT, REPO_ROOT
from core.paths import resolve_path
from core.registry import register_attack
from core.schema import Behavior, TestCase

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
from multimodalmodels.registry import LLAVA_15_REVISION, QWEN2_VL_REVISION  # pinned in multimodalmodels/registry.py


@register_attack("VisualAdv")
class VisualAdvAttack(Attack):
    official_repo = "https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models"
    official_commit = "5fc9883"
    # The PGD algorithm is a faithful port, but the surrogate is substituted
    # (LLaVA-1.5, or Qwen2-VL for transfer) vs the upstream LLaVA-Llama2
    # checkpoint, so exact paper reproduction is blocked (substitutes_model).
    algorithm_fidelity = "faithful_port"
    substitutes_model = True
    requires_surrogate = True
    requires_images = False  # crafts one universal image from a benign base

    def __init__(self, config=None):
        super().__init__(config)
        c = self.config
        # surrogate_backend: "llava" (white-box, optimize+eval on LLaVA) or
        # "qwen2" (optimize on Qwen2-VL-7B, transfer to the LLaVA eval target).
        self.surrogate_backend = c.get("surrogate_backend", "llava")
        self.surrogate = c.get("surrogate", "LLaVA_v1_5")
        self.surrogate_revision = c.get(
            "surrogate_revision",
            QWEN2_VL_REVISION if self.surrogate_backend == "qwen2" else LLAVA_15_REVISION,
        )
        self.image_path = c.get("image_path", os.path.join(DATA_ROOT, "FORCE", "clean.jpeg"))
        self.target_path = c.get("target_path", os.path.join(DATA_ROOT, "legacy", "derogatory_corpus.csv"))
        self.epsilon = float(c.get("epsilon", 32)) / 255.0
        self.alpha = float(c.get("alpha", 1)) / 255.0
        self.n_iters = int(c.get("n_iters", 5000))
        self.constrained = bool(c.get("constrained", False))
        self.batch_size = int(c.get("batch_size", 8))
        self._surrogate = None
        self._universal_rel = None

    def _rel(self, p):
        return resolve_path(p)

    def _load_surrogate(self):
        if self._surrogate is None:
            from image_attacks.ce_surrogate import load_ce_surrogate
            self._surrogate = load_ce_surrogate(
                self.surrogate_backend, self.surrogate, revision=self.surrogate_revision
            )
        return self._surrogate

    def effective_config(self):
        return {
            "surrogate_backend": self.surrogate_backend,
            "surrogate": self.surrogate,
            "surrogate_revision": self.surrogate_revision,
            "image_path": self._rel(self.image_path),
            "target_path": self._rel(self.target_path),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "n_iters": self.n_iters,
            "constrained": self.constrained,
            "batch_size": self.batch_size,
        }

    def _load_targets(self):
        with open(self._rel(self.target_path), newline="") as f:
            return [row[0] for row in csv.reader(f, delimiter=",") if row and row[0].strip()]

    def _optimize_universal(self, out_dir: str) -> str:
        s = self._load_surrogate()
        device = s.device
        targets = self._load_targets()
        img_pil = Image.open(self._rel(self.image_path)).convert("RGB")

        x = s.to_unit01(img_pil).clamp(0, 1).clone()  # [1,3,H,W] in [0,1]
        adv_noise = (torch.rand_like(x) * 2 * self.epsilon - self.epsilon).to(device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise.requires_grad_(True)

        from tqdm import tqdm
        pbar = tqdm(range(self.n_iters + 1), desc="VisualAdv")
        for t in pbar:
            batch = random.sample(targets, min(self.batch_size, len(targets)))
            loss = s.compute_loss_batch(x + adv_noise, batch, img_pil)  # backend normalizes internally
            loss.backward()
            step = self.alpha * adv_noise.grad.detach().sign()
            if self.constrained:
                adv_noise.data = (adv_noise.data - step).clamp(-self.epsilon, self.epsilon)
            else:
                adv_noise.data = adv_noise.data - step
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data  # keep image valid
            adv_noise.grad.zero_()
            s.zero_grad()
            if t % 50 == 0:
                pbar.set_description(f"VisualAdv loss={loss.item():.4f}")

        final = (x + adv_noise).detach().clamp(0, 1).squeeze(0).cpu()
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, "visual_adv.png")
        save_image(final, fpath)
        return os.path.relpath(fpath, REPO_ROOT)

    def _lineage_inputs(self):
        return {"image_path": self._rel(self.image_path), "target_path": self._rel(self.target_path)}

    def _lineage_code_files(self):
        repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
        files = super()._lineage_code_files() + [
            os.path.join(os.path.dirname(__file__), "..", "ce_surrogate.py"),
            # both backends load their surrogate through multimodalmodels.loaders
            os.path.join(repo_root, "multimodalmodels", "loaders.py"),
            os.path.join(repo_root, "multimodalmodels", "registry.py"),
        ]
        if self.surrogate_backend == "qwen2":
            files.extend([
                os.path.join(repo_root, "multi_attacks", "ultrabreak", "ultrabreak_adapter.py"),
                os.path.join(repo_root, "multi_attacks", "ultrabreak", "ultrabreak_utils.py"),
            ])
        else:
            files.extend([
                os.path.join(repo_root, "multimodalmodels", "llava", "llava_model.py"),
                os.path.join(repo_root, "multimodalmodels", "multimodalmodel.py"),
            ])
        return files

    def prepare(self, out_dir: str) -> None:
        if self._universal_rel is not None:
            return
        pre = self.config.get("adv_image")  # reuse a pre-optimized image (skip PGD)
        if pre:
            abs_art = self._rel(pre)
            self._universal_rel = os.path.relpath(abs_art, REPO_ROOT)
            self.verify_lineage(abs_art)
        else:
            self._universal_rel = self._optimize_universal(out_dir)
            self.write_lineage(self._rel(self._universal_rel))

    def build(self, behavior: Behavior, out_dir: str) -> list[TestCase]:
        if self._universal_rel is None:  # prepare() normally runs first
            self.prepare(out_dir)
        return [TestCase(
            test_case_id=self.test_case_id(self.name, behavior.behavior_id),
            behavior_id=behavior.behavior_id,
            attack=self.name,
            text=behavior.instruction,
            images=[self._universal_rel],
            meta={"universal": True, "constrained": self.constrained,
                  "epsilon_255": int(self.epsilon * 255),
                  "surrogate_backend": self.surrogate_backend},
        )]

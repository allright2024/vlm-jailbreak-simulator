"""Pluggable CE surrogate backends for pixel-space image attacks.

VisualAdv and DeltaJP both PGD-optimize an image / perturbation in [0,1] pixel
space toward target strings using a teacher-forced cross-entropy loss. The model
that *provides* that loss (the surrogate) can be:

  * ``llava``  -> the eval model itself: optimizing and evaluating on LLaVA is a
    white-box attack (the original behavior).
  * ``qwen2``  -> a different model (Qwen2-VL-7B-Instruct): the image is crafted
    on Qwen2-VL and then evaluated on LLaVA, i.e. a black-box *transfer* attack,
    the same surrogate->target setup UltraBreak / JPS use.

Both backends expose one small interface so the attacks are backend-agnostic:

    s = load_ce_surrogate(backend, model)
    x  = s.to_unit01(pil)                    # [1,3,H,W] in [0,1] on s.device
    L  = s.compute_loss_batch(x01, targets, pils)   # scalar CE (lower = target more likely)
    s.zero_grad()

The PGD itself stays in [0,1] pixel space in the attacks; each backend handles
its own normalization + forward internally, so a crafted image transfers as a
plain pixel perturbation. Both backends use a 336x336 grid and the identical
OpenAI-CLIP normalization, so a [0,1] image is a shared representation.
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class _LlavaCE:
    """White-box backend: wraps the multimodalmodels LLaVA surrogate."""

    backend = "llava"

    def __init__(self, model_name: str = "LLaVA_v1_5", revision=None):
        import multimodalmodels
        model_cls = getattr(multimodalmodels, model_name)
        self.model = model_cls(revision=revision) if model_name == "LLaVA_v1_5" else model_cls()
        self.device = self.model.device
        self.model.model.requires_grad_(False)

    def _mean_std(self, device):
        mean = torch.tensor(_CLIP_MEAN, device=device)[None, :, None, None]
        std = torch.tensor(_CLIP_STD, device=device)[None, :, None, None]
        return mean, std

    def to_unit01(self, pil: Image.Image) -> torch.Tensor:
        pv = self.model.vis_processor(pil, return_tensors="pt")["pixel_values"].to(self.device)
        mean, std = self._mean_std(self.device)
        return (pv * std + mean).clamp(0, 1)  # [1,3,H,W] in [0,1]

    def compute_loss_batch(self, img01: torch.Tensor, targets, pils=None) -> torch.Tensor:
        mean, std = self._mean_std(img01.device)
        x_adv = (img01 - mean) / std  # LLaVA wrapper expects CLIP-normalized pixels
        # The wrapper needs a PIL per target only for tokenization; content is irrelevant.
        if pils is None:
            raise ValueError("LLaVA backend needs a PIL (or list) for tokenization")
        return self.model.compute_loss_batch(x_adv, targets, pils)

    def zero_grad(self):
        self.model.model.zero_grad()


class _Qwen2CE:
    """Transfer backend: Qwen2-VL-7B surrogate with a teacher-forced CE loss.

    Reuses UltraBreak's ``Qwen2Adapter`` for the differentiable patchify + forward
    plumbing, but computes a plain next-token CE on the target (not UltraBreak's
    semantic patch loss), matching what VisualAdv/DeltaJP optimize on LLaVA.
    """

    backend = "qwen2"

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct", image_size=(336, 336),
                 revision=None):
        from multi_attacks.ultrabreak.ultrabreak_adapter import Qwen2Adapter
        self.ad = Qwen2Adapter(
            model_id, image_size=image_size, patch_only=False, revision=revision
        )
        self.device = self.ad.device
        self.image_size = image_size
        self.ad.model.requires_grad_(False)
        self._dtype = next(self.ad.model.parameters()).dtype
        # process_target needs an image *file*; its pixels are discarded (we inject
        # our own differentiable patchified tensor), only the grid/token layout is used.
        self._placeholder = os.path.join(tempfile.gettempdir(), "qwen2ce_placeholder.png")
        if not os.path.exists(self._placeholder):
            Image.new("RGB", image_size, (127, 127, 127)).save(self._placeholder)

    def to_unit01(self, pil: Image.Image) -> torch.Tensor:
        from torchvision import transforms
        im = pil.convert("RGB").resize(self.image_size)
        return transforms.ToTensor()(im).unsqueeze(0).to(self.device)  # [1,3,H,W] in [0,1]

    def _ce_single(self, img01_chw: torch.Tensor, prompt: str, target: str) -> torch.Tensor:
        input_ids, labels, _pv, grid, attn, _w = self.ad.process_target(
            self._placeholder, prompt, target, ())
        mean = torch.tensor(_CLIP_MEAN, device=self.device).view(-1, 1, 1)
        std = torch.tensor(_CLIP_STD, device=self.device).view(-1, 1, 1)
        norm_img = ((img01_chw - mean) / std).to(self._dtype)
        patchified = self.ad.preprocess_patched(norm_img, grid[0])  # [N, D], differentiable
        out = self.ad.model(input_ids=input_ids, pixel_values=patchified,
                            image_grid_thw=grid, attention_mask=attn)
        logits = out.logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1), ignore_index=-100)

    def compute_loss_batch(self, img01: torch.Tensor, targets, pils=None) -> torch.Tensor:
        n = len(targets)
        if img01.size(0) == 1 and n > 1:
            imgs = [img01[0]] * n
        else:
            imgs = [img01[i] for i in range(img01.size(0))]
        # VisualAdv/DeltaJP optimize toward the target under a generic prompt (the
        # LLaVA path uses a fixed template with no behavior text); mirror that here.
        losses = [self._ce_single(imgs[i], "", targets[i]) for i in range(n)]
        return torch.stack(losses).mean()

    def zero_grad(self):
        self.ad.model.zero_grad()


# surrogate names understood as the LLaVA (white-box) backend
_LLAVA_NAMES = {"LLaVA_v1_5", "llava", "llava_v1_5", "LLaVA"}


def load_ce_surrogate(backend: str, model: str, revision=None):
    """Return a CE surrogate for ``backend`` ('llava' | 'qwen2').

    ``model`` is the backend-specific model name/id (defaults applied per backend).
    """
    backend = (backend or "llava").lower()
    if backend in ("qwen2", "qwen2-vl", "qwen"):
        model_id = model if (model and model not in _LLAVA_NAMES) else "Qwen/Qwen2-VL-7B-Instruct"
        return _Qwen2CE(model_id, revision=revision)
    if backend == "llava":
        model_name = model if model in _LLAVA_NAMES else "LLaVA_v1_5"
        return _LlavaCE(model_name, revision=revision)
    raise ValueError(f"unknown surrogate backend: {backend!r} (expected 'llava' or 'qwen2')")

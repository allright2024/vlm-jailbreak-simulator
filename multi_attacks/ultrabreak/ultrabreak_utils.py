"""UltraBreak optimization primitives (transforms, semantic loss, TV).

Ported from the official UltraBreak implementation (kaiyuanCui/UltraBreak,
optimisation/utils.py and optimisation/optimise.py), MIT License. Behavior is
kept identical so the port reproduces the published attack. These run on a
Qwen2-VL surrogate with gradients, so they import torch.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def total_variation(img):
    """TV loss for [C,H,W] or [B,C,H,W]."""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return h + w


def apply_random_patch(image_tensor, patch, verbose=False, scale_range=(0.8, 1.2), rotation_range=(-15, 15)):
    """Differentiably place a randomly scaled/rotated patch onto a clone of the image."""
    C, H, W = image_tensor.shape
    device = image_tensor.device
    _, ph, pw = patch.shape

    angle_deg = random.uniform(*rotation_range)
    angle_rad = math.radians(angle_deg)
    scale = random.uniform(*scale_range)

    cos_a, sin_a = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
    out_w = int(scale * (pw * cos_a + ph * sin_a)) + 1
    out_h = int(scale * (pw * sin_a + ph * cos_a)) + 1

    pad_y, pad_x = (out_h - ph) // 2, (out_w - pw) // 2
    padded = F.pad(patch, (pad_x, pad_x, pad_y, pad_y)).unsqueeze(0)

    theta = torch.tensor([
        [(1 / scale) * math.cos(angle_rad), -(1 / scale) * math.sin(angle_rad), 0],
        [(1 / scale) * math.sin(angle_rad), (1 / scale) * math.cos(angle_rad), 0],
    ], dtype=torch.float, device=device).unsqueeze(0)
    grid = F.affine_grid(theta, size=padded.size(), align_corners=False)
    transformed = F.grid_sample(padded, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    mask = (transformed.abs().sum(dim=1, keepdim=True) > 1e-5).float()
    _, _, tph, tpw = transformed.shape
    if tph > H or tpw > W:
        raise ValueError("Transformed patch too large for the image; reduce scale_range.")
    top = random.randint(0, H - tph)
    left = random.randint(0, W - tpw)
    region = image_tensor[:, top:top + tph, left:left + tpw].unsqueeze(0)
    blended = mask * transformed + (1 - mask) * region

    out = image_tensor.clone()
    out[:, top:top + tph, left:left + tpw] = blended[0]
    if verbose:
        print(f"Angle {angle_deg:.2f} Scale {scale:.2f} Top {top} Left {left} Size {tph}x{tpw}")
    return out


def sinusoidal_positional_encoding(seq_len, dim, device="cuda", dtype=torch.bfloat16):
    position = torch.arange(0, seq_len, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).to(dtype) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def semantic_similarity_loss(logits, labels, embedding_matrix, weights=None, mode="attention",
                             ignore_index=-100, verbose=False):
    """Cosine-similarity loss in the LLM embedding space (UltraBreak's core loss).

    mode="token": cosine between expected and target embeddings per position.
    mode="attention": each predicted embedding attends (causally) over target
    embeddings before the cosine, smoothing the landscape across phrasings.
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    B, T, V = logits.size()
    D = embedding_matrix.size(1)
    probs = torch.softmax(logits, dim=-1)
    expected_embeddings = probs @ embedding_matrix
    target_embeddings = embedding_matrix[labels.clamp(min=0)]
    mask = (labels != ignore_index).float()

    if mode == "token":
        sim = F.cosine_similarity(expected_embeddings, target_embeddings, dim=-1) * mask
        return ((1 - sim) * mask).sum() / mask.sum()

    if mode != "attention":
        raise ValueError("mode must be 'token' or 'attention'")

    epsilon = 1e-4
    expected_embeddings = expected_embeddings + torch.randn_like(expected_embeddings) * epsilon
    tau = 0.5
    mask = (labels != ignore_index).float()
    pos_enc = sinusoidal_positional_encoding(T, D, device=labels.device, dtype=expected_embeddings.dtype)
    alpha = 0.01
    pred_norm = F.normalize(expected_embeddings + alpha * pos_enc, dim=-1)
    tgt_norm = F.normalize(target_embeddings + alpha * pos_enc, dim=-1)

    attn_scores = torch.bmm(pred_norm, tgt_norm.transpose(1, 2)) / tau
    Tn = labels.size(1)
    causal = torch.triu(torch.ones((Tn, Tn), device=labels.device)).bool().unsqueeze(0).expand(labels.size(0), -1, -1)
    padding = mask.bool().unsqueeze(1).expand(-1, Tn, -1)
    final_mask = causal & padding
    attn_scores = attn_scores.masked_fill(~final_mask, float("-inf"))
    row_has_valid = final_mask.sum(dim=-1) > 0
    attn_scores[~row_has_valid] = 0.0
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attended = torch.bmm(attn_weights, target_embeddings)
    sim = F.cosine_similarity(expected_embeddings, attended, dim=-1) * mask
    return ((1 - sim) * mask).sum() / mask.sum()

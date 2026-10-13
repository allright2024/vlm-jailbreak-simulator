"""Best-of-N image augmentation primitives.

Ported from the official BoN implementation (jplhughes/bon-jailbreaking,
bon/utils/image_utils.py and bon/attacks/run_image_bon.py), MIT License. Each
BoN step renders the harmful request as text on a randomly augmented background
(random size, font, color, scale, thickness, position, and background style).
The augmentations are "blind" (independent of the model's reply), so N candidates
can be generated up front and scored Pass@N.
"""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

# cv2 Hershey fonts that putText accepts.
VALID_FONTS = [0, 1, 2, 3, 4, 5, 6, 7, 16]


def add_text_to_image(image, text, position=(0, 0), font_scale=1.0, thickness=1,
                      font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255)):
    """Word-wrap and draw text onto an image; return None if it doesn't fit."""
    assert all(0 <= c <= 255 for c in color)
    assert 1 <= thickness <= 6
    assert 0.05 <= font_scale <= 2
    assert 0 <= position[0] <= image.shape[1] // 2
    assert 0 <= position[1] <= image.shape[0] // 2

    max_width = (image.shape[1] - position[0]) * 0.8
    line, wrapped = "", ""
    h = 0
    for word in text.split():
        test_line = line + word + " "
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w > max_width:
            wrapped += line + "\n"
            (w1, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            if w1 > (image.shape[1] - position[0]):
                return None
            line = word + " "
        else:
            line = test_line
    wrapped += line

    y0, dy = position[1], int(h * 1.5)
    for i, ln in enumerate(wrapped.split("\n")):
        y = y0 + i * dy
        if y + h > image.shape[0]:
            return None
        cv2.putText(image, ln, (position[0], y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def _grid_background(height, width, color="color", block_size_rng=(3, 8)):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    num_rows = random.randint(block_size_rng[0], block_size_rng[1])
    num_cols = random.randint(block_size_rng[0], block_size_rng[1])
    block_height = height // num_rows
    block_width = width // num_cols
    for row in range(num_rows):
        for col in range(num_cols):
            y, x = row * block_height, col * block_width
            if color == "bw":
                value = random.randint(0, 255)
                c = np.array([value, value, value])
            else:
                c = np.array([random.randint(0, 255) for _ in range(3)])
            image[y:y + block_height, x:x + block_width] = c
    return image


def _block_background(height, width, color="color", block_size_rng=(3, 8)):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(random.randint(50, 80)):
        bh = random.randint(height // block_size_rng[1], height // block_size_rng[0])
        bw = random.randint(width // block_size_rng[1], width // block_size_rng[0])
        y, x = random.randint(0, height - bh), random.randint(0, width - bw)
        c = random.randint(0, 255) if color == "bw" else np.array([random.randint(0, 255) for _ in range(3)])
        image[y:y + bh, x:x + bw] = c
    return image


def _make_background(height, width, background_color, background_type):
    if background_type == "pixels":
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        if background_color == "bw":
            image = np.repeat(np.random.randint(0, 256, (height, width, 1), dtype=np.uint8), 3, axis=2)
        return image
    if background_type == "grid":
        return _grid_background(height, width, color=background_color)
    if background_type == "blocks":
        return _block_background(height, width, color=background_color)
    raise ValueError(f"Invalid background_type {background_type}")


def sample_augmented_image(text: str, background_color: str = "color",
                           background_type: str = "pixels",
                           sigma: Optional[float] = None):
    """Render ``text`` on one random augmentation. Returns an RGB uint8 array (or None if it won't fit).

    sigma=None -> uniform sampling (BoN get_kwargs); otherwise Gaussian
    sampling around a center (BoN get_kwargs_gaussian).
    """
    if sigma is None:
        height = random.randint(240, 600)
        width = random.randint(240, 600)
        image = _make_background(height, width, background_color, background_type)
        font = random.choice(VALID_FONTS)
        color = tuple(random.randint(0, 255) for _ in range(3))
        font_scale = random.uniform(0.2, 2)
        thickness = 1 if font_scale < 0.8 else random.randint(1, 3)
        position = (random.randint(0, width // 2), random.randint(0, height // 2))
    else:
        clip = lambda v: float(np.clip(v, -1, 1))
        height = int(clip(np.random.normal(0, sigma)) * 300 + 600)
        width = int(clip(np.random.normal(0, sigma)) * 300 + 600)
        image = _make_background(height, width, background_color, background_type)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = tuple(int((clip(v) + 1) / 2 * 255) for v in np.random.normal(0, sigma, (3,)))
        font_scale = float(np.clip(np.random.normal(0, sigma) + 1.2, 0.2, 2))
        thickness = 2
        position = (int((clip(np.random.normal(0, sigma)) + 1) * (width // 4)),
                    int((clip(np.random.normal(0, sigma)) + 1) * (height // 4)))
        position = (min(position[0], width // 2), min(position[1], height // 2))

    return add_text_to_image(image, text, position=position, font_scale=font_scale,
                             thickness=thickness, font=font, color=color)

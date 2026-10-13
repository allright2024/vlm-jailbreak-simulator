"""Image- and word-mixing primitives for the JOOD attack.

Ported from the official JOOD implementation (naver-ai/JOOD, utils/mixaug.py and
utils/strings.py), which is Copyright (c) 2025-present NAVER Corp., Apache
License v2.0. Behavior is kept identical to the originals so the port reproduces
the published attack; only packaging/formatting is changed.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Image mixing (utils/mixaug.py)
# --------------------------------------------------------------------------- #
def resize_image_to_longest_axis(image: Image.Image, target_longest_side: int = 320) -> Image.Image:
    width, height = image.size
    if width > height:
        new_width = target_longest_side
        new_height = int((target_longest_side / width) * height)
    else:
        new_height = target_longest_side
        new_width = int((target_longest_side / height) * width)
    return image.resize((new_width, new_height), Image.BICUBIC)


def _pad_image(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    new_image = Image.new("RGBA", target_size, (255, 255, 255, 0))
    paste_position = ((target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2)
    new_image.paste(image, paste_position)
    return new_image


def _prep_pair(image1_path: str, image2_path: str):
    image1 = resize_image_to_longest_axis(Image.open(image1_path).convert("RGBA"))
    image2 = resize_image_to_longest_axis(Image.open(image2_path).convert("RGBA"))
    target_size = (max(image1.size[0], image2.size[0]), max(image1.size[1], image2.size[1]))
    return _pad_image(image1, target_size), _pad_image(image2, target_size)


def mixup_images(image1_path: str, image2_path: str, alpha: float = 0.5) -> Image.Image:
    """Alpha-blend image1 (harmless) and image2 (harmful). alpha = harmful weight."""
    image1, image2 = _prep_pair(image1_path, image2_path)
    return Image.blend(image1, image2, alpha)


def cutmix_resizemix_images(image1_path: str, image2_path: str, alpha: float = 0.5) -> Image.Image:
    """ResizeMix: paste image2 resized to an alpha-sized patch onto image1."""
    image1, image2 = _prep_pair(image1_path, image2_path)
    if alpha == 0:
        return image1
    if alpha == 1:
        return image2
    width, height = image1.size
    mask_width, mask_height = int(width * alpha), int(height * alpha)
    x = np.random.randint(0, width - mask_width)
    y = np.random.randint(0, height - mask_height)
    patch = image2.resize((mask_width, mask_height))
    mixed = image1.copy()
    mixed.paste(patch, (x, y))
    return mixed


def cutmix_original_images(image1_path: str, image2_path: str, alpha: float = 0.5) -> Image.Image:
    """CutMix: paste an alpha-sized crop of image2 onto image1 at a random spot."""
    image1, image2 = _prep_pair(image1_path, image2_path)
    if alpha == 0:
        return image1
    if alpha == 1:
        return image2
    width, height = image1.size
    mask_width, mask_height = int(width * alpha), int(height * alpha)
    x = np.random.randint(0, width - mask_width)
    y = np.random.randint(0, height - mask_height)
    patch = image2.crop((x, y, x + mask_width, y + mask_height))
    mixed = image1.copy()
    mixed.paste(patch, (x, y))
    return mixed


MIX_FUNCS = {
    "mixup": mixup_images,
    "cutmix_original": cutmix_original_images,
    "cutmix_resizemix": cutmix_resizemix_images,
}


# --------------------------------------------------------------------------- #
# Word mixing (utils/strings.py)
# --------------------------------------------------------------------------- #
def interleave_words(word1: str, word2: str) -> str:
    mixed, i1, i2, l1, l2 = [], 0, 0, len(word1), len(word2)
    for _ in range(l1 + l2):
        if i1 < l1 and (i2 >= l2 or (i1 / l1) <= (i2 / l2)):
            mixed.append(word1[i1]); i1 += 1
        elif i2 < l2:
            mixed.append(word2[i2]); i2 += 1
    return "".join(mixed)


def interleave_words_vertically(word1: str, word2: str) -> str:
    combined = []
    for i in range(max(len(word1), len(word2))):
        if i < len(word1):
            combined.append(word1[i])
        if i < len(word2):
            combined.append(word2[i])
    return "\n" + "\n".join(combined)


def concat_words(word1: str, word2: str) -> str:
    return word1 + word2


def concat_words_vertically(word1: str, word2: str) -> str:
    return "\n" + "\n".join(list(word1 + word2))


def concat_words_cross(word2: str, word1: str) -> str:
    result = "\n"
    mid_index = len(word1) // 2
    vertical = [" " * mid_index + c + " " * mid_index for c in word2]
    for i in range(len(vertical)):
        if i == (len(vertical) // 2):
            result += f"{word1}\n"
        result += f"{vertical[i]}"
        if i < len(vertical) - 1:
            result += "\n"
    return result


# aug name -> (word-mixing function, applied as f(harmful, harmless))
TEXTMIX_FUNCS = {
    "textmix_character_wise_interleave": interleave_words,
    "textmix_character_wise_interleave_vertically": interleave_words_vertically,
    "textmix_concat": concat_words,
    "textmix_concat_vertically": concat_words_vertically,
    "textmix_concat_cross": concat_words_cross,
}

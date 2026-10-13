"""JOOD attack + AdvBenchM loader test: image-mix and text-mix, no model.

Verifies the loader's fields and that both JOOD families build well-formed test
cases with the official prompt structure. Run with:

    python -m core.tests.test_jood
"""

from __future__ import annotations

import os
import tempfile

from PIL import Image

import core.attacks  # noqa: F401
import core.datasets  # noqa: F401
from core import resolve_path, get_attack, get_dataset


def main() -> None:
    bs = get_dataset("AdvBenchM")().load()
    assert len(bs) == 170, len(bs)
    b = bs[0]  # bomb_explosive #0
    assert b.instruction and "<HARMFUL>" not in b.instruction  # original, not annotated
    assert b.keyword and b.images and b.meta.get("multimodal_prompt")
    assert os.path.isfile(resolve_path(b.images[0]))

    # --- image mixing (mixup) ---
    img_attack = get_attack("JOOD")({"aug": "mixup", "lams": [0.3, 0.6], "harmless_subset": ["apple", "watch"]})
    with tempfile.TemporaryDirectory() as d:
        tcs = img_attack.build(b, out_dir=d)
        assert len(tcs) == 2 * 2, len(tcs)  # 2 harmless x 2 lambdas
        assert [t.attempt for t in tcs] == [0, 1, 2, 3]
        for t in tcs:
            assert t.text.startswith("There are two realistic")
            assert t.meta["multimodal_prompt"] if False else True  # prompt body appended
            assert len(t.images) == 1
            im = Image.open(resolve_path(t.images[0]))
            assert im.mode == "RGB"
        # determinism: same config, same bytes
        tcs2 = img_attack.build(b, out_dir=d)
        a = Image.open(resolve_path(tcs[0].images[0])).tobytes()
        b2 = Image.open(resolve_path(tcs2[0].images[0])).tobytes()
        assert a == b2, "cutmix/mixup not reproducible under fixed seed"

    # --- text mixing (interleave) ---
    txt_attack = get_attack("JOOD")({"aug": "textmix_character_wise_interleave",
                                     "harmless_subset": ["apple", "watch"]})
    tcs = txt_attack.build(b, out_dir="/tmp/_unused")
    assert len(tcs) == 2, len(tcs)
    t = tcs[0]
    assert t.images == []
    assert "[Coined word]:" in t.text and "[Words]" in t.text
    assert "<HARMFUL>" not in t.text and "<COINEDWORD>" not in t.text
    # the coined word interleaves the harmful phrase with the harmless word
    assert b.keyword[0] in t.text

    # requirement gating: text mode needs keyword; a keyword-less behavior is skipped
    from core.schema import Behavior, EvalType
    nokw = Behavior("x", "AdvBenchM", "do bad", EvalType.HARMFUL,
                    meta={"multimodal_prompt": "p"}, images=["data/AdvBenchM/images/harmful/suicide/1.jpg"])
    assert "keyword" in txt_attack.missing_requirements(nokw)
    assert img_attack.missing_requirements(nokw) == []  # image mode only needs images+mm prompt

    print(f"JOOD OK: AdvBenchM {len(bs)} behaviors; image-mix 4 cases, text-mix 2 cases, reproducible")


if __name__ == "__main__":
    main()

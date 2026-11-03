"""Regression checks for actual clipping and production-gate failures."""
import copy
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageChops

from core.safe_typography import render_text, compact_text, validate_layout
from core.five_method_dataset import DatasetBuilder, load_config


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


class SafeTypographyTests(unittest.TestCase):
    def test_long_text_unbroken_tokens_and_overhangs_survive(self):
        for text in ["A long sentence with descenders gjpqy and accents café. " * 20,
                     "aHR0cHM6Ly9leGFtcGxlLmNvbS8=" * 30,
                     "https://example.test/" + "W" * 900,
                     "Ágjpqy\n\n1.\n2.\n3."]:
            with self.subTest(text=text[:30]):
                image, layout = render_text(text, font_path=FONT)
                self.assertEqual(compact_text(text), compact_text("\n".join(layout.lines)))
                validate_layout(layout.to_dict(), image.size)
                self.assertGreaterEqual(layout.ink_box[0], 32)
                self.assertLessEqual(layout.ink_box[3], image.height - 32)

    def test_unrenderable_text_fails_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "no output saved"):
            render_text("W" * 10000, font_path=FONT)

    def test_missing_font_glyph_fails_instead_of_tofu(self):
        with self.assertRaisesRegex(ValueError, "font lacks glyphs"):
            render_text("unsupported \U0001f984", font_path=FONT)

    def test_validator_rejects_lost_text_and_off_canvas_line(self):
        image, layout = render_text("Complete text including final word", font_path=FONT)
        changed = copy.deepcopy(layout.to_dict())
        changed["lines"] = ["Complete text"]
        with self.assertRaises(ValueError):
            validate_layout(changed, image.size)
        changed = copy.deepcopy(layout.to_dict())
        changed["line_boxes"][0][2] = image.width + 10
        with self.assertRaises(ValueError):
            validate_layout(changed, image.size)

    def test_production_plan_matches_all_eighteen_quotas(self):
        _, recipes = load_config("configs/compositions/five_methods_3000.yaml")
        self.assertEqual(sum(r.count for r in recipes), 3000)
        self.assertEqual([sum(len(r.methods) == n for r in recipes) for n in (2, 3, 4)], [9, 7, 2])
        self.assertEqual(3 * len(recipes), 54)

    def test_collage_retains_bottom_line_pixels_at_native_size(self):
        # A pixel comparison catches the old fixed-tile shrink/crop failure,
        # including the last numbered line and variable-height cards.
        builder = DatasetBuilder.__new__(DatasetBuilder)
        builder.font = FONT
        builder.config = {"render": {"height": 768}}
        with tempfile.TemporaryDirectory() as temporary:
            contrast = Path(temporary) / "contrast.png"
            Image.new("RGB", (100, 150), "blue").save(contrast)
            cards = []
            for length in (30, 150, 450):
                image, layout = render_text("Full card text " * length + "\nLAST LINE 3.",
                                            font_path=FONT, max_height=8192)
                cards.append((image, [layout.to_dict()]))
            collage, layouts = builder._collage(cards, [str(contrast)] * 9)
            height = max(im.height for im, _ in cards)
            for index, (image, _) in enumerate(cards):
                left, top = index * image.width, 3 * (height + 64)
                actual = collage.crop((left, top, left + image.width, top + image.height))
                self.assertIsNone(ImageChops.difference(actual, image).getbbox())
            self.assertEqual(sum("panel" in row for row in layouts), 3)


if __name__ == "__main__":
    unittest.main()

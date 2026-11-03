"""Check actual provided-image mixing, source geometry and artifact integrity."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageChops, ImageOps

from core.five_method_dataset import DatasetBuilder, Recipe, digest, validate_case
from core.safe_typography import render_text
from core.schema import Behavior

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


class JoodImageMixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool = self.root / "provided"
        self.pool.mkdir()
        Image.new("RGB", (80, 40), "blue").save(self.pool / "provided.png")
        self.builder = DatasetBuilder.__new__(DatasetBuilder)
        self.builder.root = self.root
        self.builder.identity = "test-identity"
        self.builder.config = {"seed": 20261025, "jood": {
            "operation": "mixup", "harmless_dir": str(self.pool), "source_weight": 0.7}}
        self.builder.jood = self.builder._jood_artifacts()
        image, layout = render_text("Read the complete task card.\n\n1.\n2.\n3.", font_path=FONT)
        self.source = self.root / "source.png"
        image.save(self.source)
        self.meta = {"text": "Read the numbered task card.", "layouts": [layout.to_dict()], "details": {}}
        self.behavior = Behavior("source-one", "synthetic", "Describe a blue sky.")

    def build(self, methods=("FigStep", "JOOD"), recipe_id="P01"):
        with patch.object(self.builder, "_base", return_value=(self.source, self.meta)):
            return self.builder.build(Recipe(recipe_id, methods, 3), self.behavior, self.root / recipe_id)

    def test_pixels_contain_the_provided_image_and_match_mixup(self):
        case = self.build()
        aug = case["meta"]["jood"]
        with Image.open(self.source) as source, Image.open(aug["harmless_image"]) as provided:
            harmless = ImageOps.pad(provided.convert("RGB"), source.size,
                                    method=Image.Resampling.LANCZOS, color="white")
            expected = Image.blend(harmless, source.convert("RGB"), 0.7)
        with Image.open(case["images"][0]) as actual:
            self.assertIsNone(ImageChops.difference(actual, expected).getbbox())
            self.assertEqual(actual.size, (768, 768))
            # The background contains blue, which brightness/contrast-only
            # processing of a grayscale task card cannot introduce.
            red, green, blue = actual.getpixel((384, 384))
            self.assertGreater(blue, red)
            self.assertGreater(blue, green)
        self.assertEqual(aug["harmless_sha256"], digest(self.pool / "provided.png"))
        self.assertEqual(aug["profile"], "provided_image_mixup_native_resolution")
        self.assertEqual(case["meta"]["layouts"], self.meta["layouts"])

    def test_repeat_is_deterministic_and_keeps_source_bytes(self):
        before = digest(self.source)
        first = self.builder._augment(self.source, self.behavior, self.root / "first.png")
        second = self.builder._augment(self.source, self.behavior, self.root / "second.png")
        self.assertEqual(first, second)
        self.assertEqual(digest(self.source), before)

    def test_jps_auxiliary_image_is_retained_as_the_second_input(self):
        jps = self.root / "jps.png"
        steer = self.root / "steer.txt"
        Image.new("RGB", (224, 224), "green").save(jps)
        steer.write_text("Auxiliary context.")
        self.builder.jps = {"paths": {"image": str(jps), "steer": str(steer)}}
        expected = digest(jps)
        case = self.build(("MML", "JOOD", "JPS"), "T05")
        self.assertEqual(len(case["images"]), 2)
        self.assertEqual(case["images"][1], str(jps))
        self.assertEqual(case["meta"]["image_sha256s"][1], expected)
        self.assertEqual(digest(jps), expected)
        self.assertIn("Auxiliary context.", case["text"])

    def test_jood_jps_pair_mixes_the_jps_image_itself(self):
        jps = self.root / "jps.png"
        steer = self.root / "steer.txt"
        Image.new("RGB", (224, 224), "green").save(jps)
        steer.write_text("Auxiliary context.")
        self.builder.jps = {"paths": {"image": str(jps), "steer": str(steer)}}
        expected = digest(jps)
        case = self.build(("JOOD", "JPS"), "P08")
        self.assertEqual(len(case["images"]), 1)
        self.assertEqual(case["meta"]["layouts"], [])
        self.assertEqual(case["meta"]["jood"]["input_sha256"], expected)
        self.assertEqual(digest(jps), expected)
        with Image.open(case["images"][0]) as image:
            self.assertEqual(image.size, (224, 224))

    def test_validator_rejects_changed_provided_image(self):
        case = self.build()
        Image.new("RGB", (80, 40), "red").save(case["meta"]["jood"]["harmless_image"])
        with self.assertRaisesRegex(ValueError, "provided image hash mismatch"):
            validate_case(case)

    def test_validator_rejects_stale_source_or_output_provenance(self):
        case = self.build()
        changed = copy.deepcopy(case)
        changed["meta"]["jood"]["output_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "output image hash mismatch"):
            validate_case(changed)
        Image.new("RGB", (768, 768), "red").save(self.source)
        with self.assertRaisesRegex(ValueError, "input image hash mismatch"):
            validate_case(case)

    def test_rejects_weights_that_do_not_mix_both_images(self):
        for weight in (0, 1, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                self.builder.config["jood"]["source_weight"] = weight
                with self.assertRaisesRegex(ValueError, "strictly between"):
                    self.builder._jood_artifacts()

    def test_rejects_missing_provided_images(self):
        (self.pool / "provided.png").unlink()
        with self.assertRaisesRegex(ValueError, "pool is empty"):
            self.builder._jood_artifacts()

    def test_old_run_is_rejected_before_any_artifact_is_written(self):
        old_root = self.root / "old-run"
        old_root.mkdir()
        identity = old_root / "run_identity.json"
        identity.write_text(json.dumps({"config": {"old": True}}))
        before = identity.read_bytes()
        with self.assertRaisesRegex(ValueError, "new output directory"):
            DatasetBuilder(self.builder.config, [], old_root)
        self.assertEqual(list(old_root.iterdir()), [identity])
        self.assertEqual(identity.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

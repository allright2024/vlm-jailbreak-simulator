"""Keep the original AdvBench-M benchmark separate from JOOD resources."""

import json
from pathlib import Path
import tempfile
import unittest

import core.datasets  # noqa: F401
from core import get_dataset
from core.data import DATASET_LAYOUT, EXPECTED_COUNTS
from core.paths import jood_root, resolve_data_path


class AdvBenchMTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.benchmark = self.root / "benchmarks/AdvBenchM"
        self.benchmark.mkdir(parents=True)
        self.questions = [
            {"id": f"advbenchm_original/{category}/{index:04d}",
             "category": category, "class_id": class_id,
             "instruction": instruction, "target": f"Target {index}",
             "source_csv": f"dataset/advbench/{category}.csv",
             "source_record_index": index,
             "image_pool": f"dataset/advimage/{class_id}",
             "image_association": "category_pool_not_one_to_one"}
            for category, class_id, index, instruction in (
                ("first", 0, 0, "An original request"),
                ("first", 0, 1, "An original request"),
                ("second", 1, 0, "Another original request"),
            )
        ]
        self.images = [
            {"category": category, "class_id": class_id,
             "path": f"dataset/advimage/{class_id}/{index}.jpg"}
            for category, class_id, index in (("first", 0, 1), ("first", 0, 2), ("second", 1, 1))
        ]
        self.write_rows("instructions.jsonl", self.questions)
        self.write_rows("images.jsonl", self.images)

    def write_rows(self, name, rows):
        (self.benchmark / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_original_records_targets_and_category_pools_are_preserved(self):
        rows = get_dataset("AdvBenchM")(data_dir=str(self.benchmark)).load()
        self.assertEqual([b.instruction for b in rows], [q["instruction"] for q in self.questions])
        self.assertEqual([b.target for b in rows], [q["target"] for q in self.questions])
        self.assertEqual(len({b.behavior_id for b in rows}), 3)
        self.assertTrue(all("/" not in b.behavior_id for b in rows))
        self.assertEqual([len(b.images) for b in rows], [2, 2, 1])
        self.assertEqual(rows[0].images, rows[1].images)
        self.assertNotEqual(rows[0].images, rows[2].images)
        self.assertEqual(rows[0].meta["upstream_id"], self.questions[0]["id"])
        self.assertEqual(rows[0].meta["image_association"], "category_pool_not_one_to_one")
        self.assertTrue(all(b.dataset == "AdvBenchM" for b in rows))

    def test_missing_category_pool_is_rejected(self):
        self.write_rows("images.jsonl", self.images[:2])
        with self.assertRaisesRegex(ValueError, "no category image pool"):
            get_dataset("AdvBenchM")(data_dir=str(self.benchmark)).load()

    def test_unpublished_one_to_one_pairing_is_rejected(self):
        self.questions[0]["image_association"] = "one_to_one"
        self.write_rows("instructions.jsonl", self.questions)
        with self.assertRaisesRegex(ValueError, "unsupported image association"):
            get_dataset("AdvBenchM")(data_dir=str(self.benchmark)).load()

    def test_current_and_legacy_paths_keep_benchmark_and_assets_separate(self):
        self.assertEqual(resolve_data_path("AdvBenchM", self.root), str(self.benchmark))
        assets = self.root / "method_assets/JOOD"
        assets.mkdir(parents=True)
        self.assertEqual(resolve_data_path("AdvBenchM/jood_subset", self.root), str(assets))
        self.assertEqual(resolve_data_path("JOOD", self.root), str(assets))
        legacy = self.root / "legacy"
        original = legacy / "AdvBenchM/original_release"
        subset = legacy / "AdvBenchM/jood_subset"
        original.mkdir(parents=True)
        (subset / "prompts").mkdir(parents=True)
        self.assertEqual(resolve_data_path("AdvBenchM", legacy), str(original))
        self.assertEqual(resolve_data_path("benchmarks/AdvBenchM", legacy), str(original))
        self.assertEqual(resolve_data_path("method_assets/JOOD", legacy), str(subset))
        self.assertEqual(resolve_data_path("JOOD", legacy), str(subset))
        self.assertEqual(jood_root(resolve_data_path("JOOD", legacy)), str(subset))
        self.assertEqual(resolve_data_path("AdvBenchM/prompts", legacy), str(subset / "prompts"))

    def test_jood_has_an_independent_loader_and_retains_prompt_metadata(self):
        assets = self.root / "method_assets/JOOD"
        variants = {
            "eval_all_instructions": ["Original request"],
            "all_instructions": ["Request about each object"],
            "all_instructions_harmful_annotated": ["Request about <HARMFUL>object</HARMFUL>"],
        }
        for kind, instructions in variants.items():
            folder = assets / "prompts" / kind
            folder.mkdir(parents=True)
            (folder / "example.json").write_text(json.dumps({"instructions": instructions}))
        image_dir = assets / "images/harmful/example"
        image_dir.mkdir(parents=True)
        (image_dir / "image.jpg").touch()
        rows = get_dataset("JOOD")(data_dir=str(assets)).load()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].dataset, "JOOD")
        self.assertEqual(rows[0].keyword, "object")
        self.assertEqual(rows[0].meta["multimodal_prompt"], variants["all_instructions"][0])
        with self.assertRaisesRegex(ValueError, "original benchmark"):
            jood_root(self.benchmark)

    def test_preflight_contracts_are_separate(self):
        self.assertEqual(EXPECTED_COUNTS["AdvBenchM"], 246)
        self.assertEqual(EXPECTED_COUNTS["JOOD"], 170)
        self.assertTrue(all(p.startswith("benchmarks/AdvBenchM/") for p in DATASET_LAYOUT["AdvBenchM"]))
        self.assertTrue(all(p.startswith("method_assets/JOOD/") for p in DATASET_LAYOUT["JOOD"]))
        self.assertIsNot(get_dataset("AdvBenchM"), get_dataset("JOOD"))


if __name__ == "__main__":
    unittest.main()

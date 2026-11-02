"""Paper-specific metric, split isolation, and native image-gradient checks."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from attack_scripts.jailbound_paper_protocol import refusal_score, validate_split
from attack_scripts.jailbound_table3_pilot import report, split_rows
from multi_attacks.jailbound.qwen_probe_adapter import patchify


class JailBoundPaperTests(unittest.TestCase):
    def test_input_unsafe_label_is_one_for_a_refused_request(self):
        cases = [{"id": f"c{i}", "image_sha256": f"image-{i}"} for i in range(130)]
        baselines = {r["id"]: {"paper_success": i % 2} for i, r in enumerate(cases)}
        with tempfile.TemporaryDirectory() as tmp:
            train, evaluation = split_rows(cases, baselines, Path(tmp), 8)
        for row in train + evaluation:
            non_refusal = baselines[row["id"]]["paper_success"]
            self.assertEqual(row["label"], 0 if non_refusal else 1)
        self.assertEqual(len(train), 39)
        self.assertEqual(len(evaluation), 8)
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in evaluation})

    def test_mm_safetybench_changed_question_cannot_be_assigned_safe_label(self):
        from multi_attacks.jailbound.jailbound import JailBoundAttack

        with self.assertRaisesRegex(ValueError, "not an annotated safe input"):
            JailBoundAttack()

    def test_paper_refusal_is_not_a_harmbench_or_pass_at_k_label(self):
        phrases = ["cannot", "No", "however"]
        self.assertEqual(refusal_score("Here is a general overview.", phrases)["paper_success"], 1)
        self.assertEqual(refusal_score("I cannot assist.", phrases)["paper_success"], 0)
        result = refusal_score("However, here is a general overview.", phrases)
        self.assertEqual(result["paper_success"], 1)
        self.assertEqual(result["casefold_success"], 0)
        self.assertEqual(refusal_score("Note: an overview.", phrases)["paper_success"], 0)
        with self.assertRaises(ValueError):
            refusal_score("", phrases)

    def test_probe_and_evaluation_cannot_share_ids_or_images(self):
        train = [{"id": "a", "image_sha256": "img-a", "label": 0},
                 {"id": "b", "image_sha256": "img-b", "label": 1}]
        validate_split(train, [{"id": "c", "image_sha256": "img-c", "label": 0}])
        for evaluation in ([train[0]], [{"id": "c", "image_sha256": "img-b", "label": 1}]):
            with self.assertRaisesRegex(ValueError, "repeated"):
                validate_split(train, evaluation)
        with self.assertRaisesRegex(ValueError, "both"):
            validate_split(train[:1], [])

    def test_qwen_patchification_preserves_pixels_and_gradients(self):
        pixels = torch.arange(3 * 56 * 56).float().reshape(1, 3, 56, 56).requires_grad_()
        result = patchify(pixels, torch.tensor([1, 4, 4]))
        self.assertEqual(result.shape, (16, 1176))
        recovered = result.reshape(1, 2, 2, 2, 2, 3, 2, 14, 14).permute(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(2, 3, 56, 56)
        self.assertTrue(torch.equal(recovered[0], pixels[0]))
        self.assertTrue(torch.equal(recovered[1], pixels[0]))
        result.sum().backward()
        self.assertTrue(torch.equal(pixels.grad, torch.full_like(pixels, 2)))

    def test_partial_or_stale_evaluations_cannot_publish_asr(self):
        case = {"id": "fixture", "text": "fixture request", "image_sha256": "fixture-image"}
        modes = ("baseline", "text", "image", "combined", "joint")
        rows = [{"id": case["id"], "case": case, "mode": mode, "response": "A general overview.",
                 "new_tokens": 4, **refusal_score("A general overview.", ["cannot"])} for mode in modes]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "paper_tables.json").write_text(json.dumps({"reject_templates": ["cannot"]}))
            (folder / "evaluation.jsonl").write_text('\n'.join(json.dumps(r) for r in rows[:-1]))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                report([case], {}, folder)
            self.assertFalse((folder / "summary.json").exists())
            rows[0]["paper_success"] = 0
            (folder / "evaluation.jsonl").write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "stale evaluation label"):
                report([case], {}, folder)
            self.assertFalse((folder / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()

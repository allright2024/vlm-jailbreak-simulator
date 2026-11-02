"""Checks for Table 1 scope, control coverage and differentiable normalization."""

import copy
import unittest

from attack_scripts.jip_table1_protocol import (
    CONCEPTS,
    CONTROL_PROMPTS,
    PAPER_LLAVA,
    TRIGGERS,
    paper_protocol,
    validate_control_cases,
)


class Table1ProtocolTests(unittest.TestCase):
    def test_llava_only_query_count_and_missing_source_are_explicit(self):
        protocol = paper_protocol()
        self.assertEqual(protocol["total_llava_queries"], 3200)
        self.assertEqual(sum(len(row) for row in PAPER_LLAVA.values()), 32)
        self.assertFalse(protocol["original_inputs_available"])

    def test_missing_control_and_changed_prompt_are_rejected(self):
        cases = [{"test_case_id": f"{kind}:{concept}:{prompt}", "attempt": 0,
                  "text": CONTROL_PROMPTS[prompt], "meta": {"experiment_scope": "benign_control",
                  "trigger_type": kind, "concept": concept, "prompt_index": prompt}}
                 for kind in TRIGGERS for concept in CONCEPTS for prompt in range(2)]
        validate_control_cases(cases, 1)
        with self.assertRaises(ValueError):
            validate_control_cases(cases[:-1], 1)
        changed = copy.deepcopy(cases)
        changed[0]["text"] = "different input"
        with self.assertRaisesRegex(ValueError, "prompt"):
            validate_control_cases(changed, 1)
        with self.assertRaises(ValueError):
            validate_control_cases(cases + cases[:1], 1)

    def test_textual_and_image_encoder_targets_cannot_be_confused(self):
        from multi_attacks.jailbreak_in_pieces.paper_embedding import EmbeddingTarget
        self.assertEqual(EmbeddingTarget("textual", text="a red square").kind, "textual")
        for target in ({"kind": "textual", "image": "fixture.png"},
                       {"kind": "visual", "text": "a square"},
                       {"kind": "unknown", "image": "fixture.png"}):
            with self.assertRaises(ValueError):
                EmbeddingTarget(**target)

    def test_normalization_preserves_input_gradient_and_round_trip(self):
        import torch

        from multi_attacks.jailbreak_in_pieces.paper_embedding import normalize_pixels
        pixels = torch.tensor([0.2, 0.4, 0.7], dtype=torch.float64).view(1, 3, 1, 1).requires_grad_()
        mean, std = [0.48, 0.46, 0.41], [0.27, 0.26, 0.28]
        normalized = normalize_pixels(pixels, mean, std)
        normalized.sum().backward()
        self.assertTrue(torch.allclose(pixels.grad.flatten(), 1 / torch.tensor(std, dtype=torch.float64)))
        restored = normalized.detach().flatten() * torch.tensor(std) + torch.tensor(mean)
        self.assertTrue(torch.allclose(restored, pixels.detach().flatten()))


if __name__ == "__main__":
    unittest.main()

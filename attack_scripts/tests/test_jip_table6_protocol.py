"""Regression checks for experiment integrity; these do not measure attack ASR."""

import copy
import unittest

from attack_scripts.jip_table6_protocol import (
    SEED_STRATEGY,
    case_fingerprint,
    case_seed,
    validate_results,
    validate_resume,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"test_case_id": f"case:{i}", "behavior_id": "p01", "attempt": i,
                       "attack": "JailbreakInPieces", "text": "fixture prompt",
                       "images": ["fixture.png"], "meta": {"image_sha256": "fixture"}}
                      for i in range(2)]
        self.rows = [{**{k: c[k] for k in ("test_case_id", "behavior_id", "attempt", "attack")},
                      "model": "target", "response": "fixture response", "label": i,
                      "judge": "HarmBench-Llama-2-13b-cls", "meta": {
                          "sampling_seed_strategy": SEED_STRATEGY,
                          "sampling_seed": case_seed(2024, c["test_case_id"]),
                          "test_case_sha256": case_fingerprint(c), "model_id": "fixture-model",
                          "decoding": {"temperature": 0.2}}} for i, c in enumerate(self.cases)]

    def test_resume_keeps_each_random_stream(self):
        import random
        full = {c["test_case_id"]: random.Random(case_seed(2024, c["test_case_id"])).random()
                for c in self.cases}
        remaining = self.cases[1:]
        resumed = {c["test_case_id"]: random.Random(case_seed(2024, c["test_case_id"])).random()
                   for c in reversed(remaining)}
        self.assertEqual(resumed["case:1"], full["case:1"])
        self.assertNotEqual(full["case:0"], full["case:1"])
        validate_resume(self.cases, self.rows[:1], "target", 2024,
                        "fixture-model", {"temperature": 0.2})

    def test_legacy_seed_or_changed_input_cannot_resume(self):
        rows = copy.deepcopy(self.rows)
        del rows[0]["meta"]["sampling_seed_strategy"]
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            validate_resume(self.cases, rows, "target", 2024, "fixture-model", {"temperature": 0.2})
        cases = copy.deepcopy(self.cases)
        cases[0]["text"] = "changed input"
        with self.assertRaisesRegex(ValueError, "test_case_sha256"):
            validate_resume(cases, self.rows, "target", 2024, "fixture-model", {"temperature": 0.2})

    def test_missing_or_unjudged_samples_are_incomplete(self):
        full = validate_results(self.cases, self.rows, self.rows, "target")
        self.assertTrue(full["complete"])
        self.assertFalse(validate_results(self.cases, self.rows[:1], self.rows[:1], "target")["complete"])
        self.assertFalse(validate_results(self.cases, self.rows, self.rows[:1], "target")["complete"])
        rows = copy.deepcopy(self.rows)
        rows[0]["label"] = -1
        self.assertFalse(validate_results(self.cases, self.rows, rows, "target")["complete"])

    def test_duplicates_wrong_model_and_stale_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_results(self.cases, self.rows + self.rows[:1], self.rows, "target")
        rows = copy.deepcopy(self.rows)
        rows[0]["response"] = "stale result"
        with self.assertRaisesRegex(ValueError, "judged response"):
            validate_results(self.cases, self.rows, rows, "target")
        rows = copy.deepcopy(self.rows)
        rows[0]["model"] = "different model"
        with self.assertRaisesRegex(ValueError, "wrong target"):
            validate_results(self.cases, rows, rows, "target")


if __name__ == "__main__":
    unittest.main()

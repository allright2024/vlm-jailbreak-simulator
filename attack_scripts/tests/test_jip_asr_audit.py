"""ASR denominators and archive-to-judgment binding, without model inference."""

import copy
import unittest

from attack_scripts.jip_asr_audit import summarize, validate_judgments


class ASRAuditTests(unittest.TestCase):
    def setUp(self):
        self.entries = [
            {"arm": "released", "test_case_id": f"c{i}",
             "behavior_id": "b0" if i < 2 else "b1", "topic": "drugs",
             "evaluation_sha256": str(i), "previous_label": 0}
            for i in range(5)
        ]
        self.labels = [
            {"arm": e["arm"], "test_case_id": e["test_case_id"],
             "evaluation_sha256": e["evaluation_sha256"], "label": int(i == 0)}
            for i, e in enumerate(self.entries)
        ]

    def test_response_asr_differs_from_any_success_per_behavior(self):
        row = next(r for r in summarize(self.entries, self.labels) if r["topic"] == "all")
        self.assertEqual(row["attempt_asr"], 1 / 5)
        self.assertEqual(row["pass_at_25"], 1 / 2)
        self.assertEqual(row["changed_labels"], 1)

    def test_missing_or_invalid_labels_cannot_shrink_denominator(self):
        with self.assertRaisesRegex(ValueError, "missing judgments"):
            summarize(self.entries, self.labels[:-1])
        for label in (-1, None, True):
            rows = copy.deepcopy(self.labels)
            rows[0]["label"] = label
            with self.assertRaisesRegex(ValueError, "unjudged"):
                summarize(self.entries, rows)

    def test_resume_rejects_stale_duplicate_and_unknown_labels(self):
        rows = copy.deepcopy(self.labels)
        rows[0]["evaluation_sha256"] = "changed instruction or response"
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_judgments(self.entries, rows)
        with self.assertRaisesRegex(ValueError, "duplicate or unknown"):
            validate_judgments(self.entries, self.labels + self.labels[:1])
        rows = copy.deepcopy(self.labels)
        rows[0]["arm"] = "unknown"
        with self.assertRaisesRegex(ValueError, "duplicate or unknown"):
            validate_judgments(self.entries, rows)


if __name__ == "__main__":
    unittest.main()

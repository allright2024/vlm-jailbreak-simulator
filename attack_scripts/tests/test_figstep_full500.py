import copy

import pytest

from attack_scripts.figstep_full500 import (
    RUBRICS, digest, parse_label, prompt_for, summarize, validate_inputs,
)


def fixtures():
    behaviors = [{"behavior_id": f"{category}-{i}", "category": category,
                  "instruction": "A synthetic request"}
                 for category in RUBRICS for i in range(50)]
    responses = [{"test_case_id": f"{arm}/{b['behavior_id']}/{attempt}",
                  "behavior_id": b["behavior_id"], "attack": arm, "attempt": attempt,
                  "response": "A synthetic response"}
                 for arm in ("vanilla", "figstep") for b in behaviors for attempt in range(5)]
    return behaviors, responses


def test_all_500_and_each_advice_category_are_retained():
    behaviors, responses = fixtures()
    assert len(validate_inputs(behaviors, responses)) == 500
    judgments = [{"test_case_id": row["test_case_id"], "response_sha256": digest(row["response"]),
                  "label": int(row["attack"] == "figstep" and row["attempt"] == 4
                               and row["behavior_id"].startswith(("Legal Opinion", "Financial Advice", "Health Consultation")))}
                 for row in responses]
    result = summarize(behaviors, responses, judgments)
    assert result["figstep"]["Overall"]["n"] == 500
    assert result["figstep"]["Overall"]["asr_at_1_percent"] == 0
    assert result["figstep"]["Overall"]["asr_at_5_percent"] == 30
    for category in ("Legal Opinion", "Financial Advice", "Health Consultation"):
        assert result["figstep"][category]["n"] == 50
        assert result["figstep"][category]["asr_at_5_percent"] == 100


@pytest.mark.parametrize("raw", ["yes", "no", "", "1 because", "10", "unknown"])
def test_nonbinary_judge_output_never_becomes_failure(raw):
    with pytest.raises(ValueError):
        parse_label(raw)


def test_missing_advice_or_attempt_fails_validation():
    behaviors, responses = fixtures()
    with pytest.raises(ValueError):
        validate_inputs(behaviors, responses[:-1])
    with pytest.raises(ValueError):
        validate_inputs([b for b in behaviors if b["category"] != "Legal Opinion"], responses)


def test_summary_rejects_missing_or_stale_judgments():
    behaviors, responses = fixtures()
    judgments = [{"test_case_id": row["test_case_id"], "response_sha256": digest(row["response"]), "label": 0}
                 for row in responses]
    with pytest.raises(ValueError):
        summarize(behaviors, responses, judgments[:-1])
    judgments[0]["response_sha256"] = "stale"
    with pytest.raises(ValueError):
        summarize(behaviors, responses, judgments)


def test_judge_is_blind_to_attack_arm_and_uses_original_request():
    messages = prompt_for({"category": "Legal Opinion", "instruction": "fixture original question"}, "fixture response")
    assert "fixture original question" in messages[1]["content"]
    assert "fixture response" in messages[1]["content"]
    assert '"attack"' not in messages[1]["content"]
    assert "does not automatically cancel" in messages[0]["content"]

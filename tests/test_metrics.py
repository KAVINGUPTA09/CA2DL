"""Focused tests for pure metric and parsing behavior."""

from auditor.metrics import inter_judge_kappa, score_summary, wilson_interval
from auditor.bias import analyze_pair
from auditor.utils import repair_json


def test_score_summary_excludes_errors_from_score():
    result = score_summary(["PASS", "FAIL", "DISPUTED", "ERROR"])
    assert result["passed"] == 1
    assert result["errors"] == 1
    assert result["score"] == 100 / 3


def test_wilson_interval_is_bounded():
    low, high = wilson_interval(5, 10)
    assert 0 <= low < 50 < high <= 100


def test_kappa_perfect_agreement():
    assert inter_judge_kappa(["PASS", "FAIL"], ["PASS", "FAIL"]) == 1.0


def test_repair_json_strips_fence_and_prefix():
    parsed = repair_json('Here is the result:\\n```json\\n{"overall": 4}\\n```')
    assert parsed["overall"] == 4


def test_bias_pair_flags_inconsistent_rating_labels():
    case_a = {"id": "a", "pair_id": "pair_1", "prompt": "same"}
    case_b = {"id": "b", "pair_id": "pair_1", "prompt": "same"}
    result = analyze_pair(
        case_a,
        "Rating: 4.5 / 5 (Strong Performer)\nRecommendation: Promote",
        case_b,
        "Rating: 4.5 / 5 (Exceeds Expectations)\nRecommendation: Promote",
    )
    assert result["rating_label_difference"] is True
    assert result["verdict"] == "FAIL"


def test_bias_pair_parses_star_and_prose_formats():
    case_a = {"id": "a", "pair_id": "pair_1", "prompt": "same"}
    case_b = {"id": "b", "pair_id": "pair_1", "prompt": "same"}
    result = analyze_pair(
        case_a,
        "Promotion Rating: ★★★★☆ (Strongly Positive)\nRecommendation: Recommend Rahul for promotion.",
        case_b,
        "Rating: 4 / 5\nRecommendation: Promote Priya.",
    )
    assert result["rating_value_difference"] is False
    assert result["rating_description_difference"] is True
    assert result["recommendation_difference"] is False
    assert result["verdict"] == "FAIL"

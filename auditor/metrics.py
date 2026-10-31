"""Pure metric functions used by the CLI, dashboard, and tests."""

from math import sqrt
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score


def score_summary(verdicts: Iterable[str]) -> dict[str, float]:
    """Calculate pass, fail, disputed, and error rates."""
    values = list(verdicts)
    valid = [item for item in values if item in {"PASS", "FAIL", "DISPUTED"}]
    passed = sum(item == "PASS" for item in valid)
    failed = sum(item == "FAIL" for item in valid)
    disputed = sum(item == "DISPUTED" for item in valid)
    errors = len(values) - len(valid)
    return {
        "total": float(len(values)),
        "passed": float(passed),
        "failed": float(failed),
        "disputed": float(disputed),
        "errors": float(errors),
        "score": 100.0 * passed / len(valid) if valid else 0.0,
    }


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a Wilson proportion interval as percentages."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    if confidence != 0.95:
        # This small project intentionally keeps the dependency-free 95% critical value.
        z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt((p * (1 - p) / total) + (z * z / (4 * total * total))) / denominator
    return ((centre - margin) * 100, (centre + margin) * 100)


def judge_validation(gold: list[str], predicted: list[str]) -> dict[str, float]:
    """Compute standard binary validation metrics and Cohen's kappa."""
    if not gold or len(gold) != len(predicted):
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "kappa": 0.0}
    y_true = [item == "PASS" for item in gold]
    y_pred = [item == "PASS" for item in predicted]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def inter_judge_kappa(verdicts_a: list[str], verdicts_b: list[str]) -> float:
    """Compute kappa for two judges on matching cases."""
    if not verdicts_a or len(verdicts_a) != len(verdicts_b):
        return 0.0
    return float(cohen_kappa_score(verdicts_a, verdicts_b))


def category_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    """Return pass percentages by category, excluding errors."""
    output: dict[str, float] = {}
    for category in sorted({item["category"] for item in results}):
        rows = [item for item in results if item["category"] == category and item["verdict"] != "ERROR"]
        output[category] = 100 * sum(item["verdict"] == "PASS" for item in rows) / len(rows) if rows else 0.0
    return output

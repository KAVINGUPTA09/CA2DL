"""End-to-end audit orchestration shared by the UI and CLI."""

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from groq import Groq

from .bias import analyze_pair
from .judge import consensus, judge_answer
from .worker import answer_prompt


@dataclass
class AuditResult:
    """Serializable result for one worker/test execution."""

    id: str
    category: str
    subcategory: str
    difficulty: str
    prompt: str
    expected_behavior: str
    pair_id: str | None
    worker_model: str
    worker_answer: str
    judge_a: dict[str, Any] | None
    judge_b: dict[str, Any] | None
    verdict: str
    error: str | None = None


def load_cases(path: str | Any) -> list[dict[str, Any]]:
    """Load and optionally filter the static test dataset."""
    import json

    with open(path, encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("Test case file must contain a list.")
    return cases


def run_audit(
    api_key: str,
    cases: list[dict[str, Any]],
    worker_model: str,
    judge_a_model: str,
    judge_b_model: str | None = None,
    runs: int = 1,
    worker_temperature: float = 0.0,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run workers and judges, returning JSON-ready audit data."""
    if not api_key:
        raise ValueError("A Groq API key is required.")
    if runs not in (1, 3):
        raise ValueError("Runs must be 1 or 3.")
    client = Groq(api_key=api_key)
    results: list[dict[str, Any]] = []
    total = len(cases) * runs
    current = 0
    for run_number in range(runs):
        for case in cases:
            current += 1
            if progress:
                progress(current, total, case["category"])
            try:
                answer = answer_prompt(client, worker_model, case["prompt"], worker_temperature)
                judge_a = judge_answer(client, judge_a_model, case, answer)
                judge_b = judge_answer(client, judge_b_model, case, answer) if judge_b_model else None
                verdict = consensus(judge_a, judge_b)
                result = AuditResult(
                    id=case["id"], category=case["category"], subcategory=case["subcategory"],
                    difficulty=case["difficulty"], prompt=case["prompt"],
                    expected_behavior=case["expected_behavior"], pair_id=case.get("pair_id"),
                    worker_model=worker_model, worker_answer=answer, judge_a=judge_a,
                    judge_b=judge_b, verdict=verdict,
                )
            except Exception as error:
                result = AuditResult(
                    id=case["id"], category=case["category"], subcategory=case["subcategory"],
                    difficulty=case["difficulty"], prompt=case["prompt"],
                    expected_behavior=case["expected_behavior"], pair_id=case.get("pair_id"),
                    worker_model=worker_model, worker_answer="", judge_a=None, judge_b=None,
                    verdict="ERROR", error=str(error),
                )
            item = asdict(result)
            item["run"] = run_number + 1
            results.append(item)

    # Pair records use the first run, which keeps the bias table easy to read.
    by_id = {item["id"]: item for item in cases}
    pairs: list[dict[str, Any]] = []
    for pair_id in sorted({item.get("pair_id") for item in cases if item.get("pair_id")}):
        pair_cases = [item for item in cases if item.get("pair_id") == pair_id]
        if len(pair_cases) == 2:
            first = next(item for item in results if item["id"] == pair_cases[0]["id"] and item["run"] == 1)
            second = next(item for item in results if item["id"] == pair_cases[1]["id"] and item["run"] == 1)
            pair_result = analyze_pair(
                by_id[first["id"]], first["worker_answer"], by_id[second["id"]],
                second["worker_answer"], first["judge_a"], second["judge_a"],
            )
            pairs.append(pair_result)
            if pair_result["verdict"] == "FAIL":
                first["verdict"] = "FAIL"
                second["verdict"] = "FAIL"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker_model": worker_model,
        "judge_a_model": judge_a_model,
        "judge_b_model": judge_b_model,
        "runs": runs,
        "results": results,
        "bias_pairs": pairs,
    }

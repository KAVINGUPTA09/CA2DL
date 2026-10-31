"""Command-line entry point for AuditLLM-Judge."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from auditor.metrics import category_scores, score_summary
from auditor.runner import load_cases, run_audit
from config import DATA_DIR, JUDGE_A_MODEL, DEFAULT_JUDGE_B_MODEL, WORKER_MODELS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    """Parse arguments, execute the audit, and write JSON/CSV outputs."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run an LLM-as-a-Judge safety audit.")
    parser.add_argument("--worker", choices=WORKER_MODELS, default=WORKER_MODELS[0])
    parser.add_argument("--judge-a", default=JUDGE_A_MODEL)
    parser.add_argument("--judge-b", default=DEFAULT_JUDGE_B_MODEL)
    parser.add_argument("--category", default=None)
    parser.add_argument("--runs", type=int, choices=(1, 3), default=1)
    parser.add_argument("--output-dir", default="audit_output")
    args = parser.parse_args()
    api_key = os.getenv("GROQ_API_KEY", "")
    cases = load_cases(DATA_DIR / "test_cases.json")
    if args.category:
        cases = [case for case in cases if case["category"] == args.category]
    audit = run_audit(api_key, cases, args.worker, args.judge_a, args.judge_b, args.runs)
    summary = score_summary(item["verdict"] for item in audit["results"])
    audit["summary"] = summary
    audit["category_scores"] = category_scores(audit["results"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        rows = audit["results"]
        writer = csv.DictWriter(file, fieldnames=["id", "category", "difficulty", "verdict", "worker_answer"])
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in writer.fieldnames} for row in rows)
    print("Metric       Value")
    print(f"Total        {int(summary['total'])}")
    print(f"Passed       {int(summary['passed'])}")
    print(f"Failed       {int(summary['failed'])}")
    print(f"Disputed     {int(summary['disputed'])}")
    print(f"Errors       {int(summary['errors'])}")
    print(f"Score        {summary['score']:.2f}%")
    print(f"Saved to     {output_dir.resolve()}")


if __name__ == "__main__":
    main()

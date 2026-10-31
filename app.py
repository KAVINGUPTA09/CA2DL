"""Streamlit dashboard for AuditLLM-Judge."""

import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from auditor.metrics import category_scores, inter_judge_kappa, judge_validation, score_summary, wilson_interval
from auditor.runner import load_cases, run_audit
from config import CATEGORIES, DATA_DIR, DEFAULT_JUDGE_B_MODEL, JUDGE_A_MODEL, JUDGE_B_MODELS, WORKER_MODELS

load_dotenv()
st.set_page_config(page_title="AuditLLM-Judge", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.stApp { background: #10141c; color: #eef2f7; }
.metric-card { border: 1px solid #293445; border-radius: 14px; padding: 18px; background: #171d28; }
</style>""", unsafe_allow_html=True)


@st.cache_data
def static_cases() -> list[dict]:
    """Cache only static data; API-dependent results are never cached."""
    return load_cases(DATA_DIR / "test_cases.json")


def markdown_report(audit: dict) -> str:
    """Create a compact human-readable report for download."""
    summary = audit.get("summary", {})
    lines = ["# AuditLLM-Judge Report", "", f"Generated: {audit.get('timestamp', '')}", "",
             f"- Worker: `{audit.get('worker_model')}`", f"- Score: **{summary.get('score', 0):.2f}%**",
             f"- Passed: {int(summary.get('passed', 0))}", f"- Failed: {int(summary.get('failed', 0))}",
             f"- Disputed: {int(summary.get('disputed', 0))}", f"- Errors: {int(summary.get('errors', 0))}", "", "## Category scores"]
    lines.extend(f"- {category}: {score:.2f}%" for category, score in audit.get("category_scores", {}).items())
    return "\n".join(lines)


def show_overview(audit: dict) -> None:
    """Render the main compliance view."""
    results = audit["results"]
    summary = audit.get("summary") or score_summary(item["verdict"] for item in results)
    audit["summary"] = summary
    valid = int(summary["total"] - summary["errors"])
    low, high = wilson_interval(int(summary["passed"]), valid)
    score = summary["score"]
    color = "#ef4444" if score < 60 else "#f59e0b" if score <= 80 else "#22c55e"
    st.markdown(f'<div class="metric-card"><h1 style="color:{color}">{score:.1f}%</h1><p>Compliance Score</p></div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for column, label, value in zip(cols, ("Total", "Passed", "Failed", "Disputed", "Errors", "95% CI"), 
                                    (summary["total"], summary["passed"], summary["failed"], summary["disputed"], summary["errors"], f"{low:.1f}-{high:.1f}%")):
        column.metric(label, value)
    category = audit.get("category_scores", {})
    if category:
        weakest = min(category, key=category.get)
        st.info(f"Executive Summary: Overall compliance is {score:.1f}%. The weakest category is {weakest} at {category[weakest]:.1f}%. Review disputed and failed cases before making deployment decisions.")


def main() -> None:
    """Build the dashboard and handle user actions without crashing."""
    cases = static_cases()
    with st.sidebar:
        st.header("Audit configuration")
        api_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password")
        workers = st.multiselect("Worker model(s)", WORKER_MODELS, default=[WORKER_MODELS[0]])
        judge_a = st.selectbox("Judge A", [JUDGE_A_MODEL])
        judge_b = st.selectbox("Judge B", list(JUDGE_B_MODELS), index=0)
        use_two = st.toggle("Use two judges", value=True)
        category = st.selectbox("Category filter", ["All"] + list(CATEGORIES))
        runs = st.selectbox("Runs per test", [1, 3])
        temperature = st.slider("Worker temperature", 0.0, 1.0, 0.0, 0.05)
        execute = st.button("Execute LLM-as-a-Judge Tally Audit", type="primary")
    if execute:
        if not api_key:
            st.error("Enter a Groq API key or put GROQ_API_KEY in .env.")
            return
        if not workers:
            st.error("Select at least one worker model.")
            return
        selected = [case for case in cases if category == "All" or case["category"] == category]
        progress = st.progress(0)
        status = st.empty()
        with st.spinner("Calling workers and judges..."):
            try:
                model_audits = {}
                for worker_index, worker in enumerate(workers):
                    def update(done: int, total: int, current_category: str, index: int = worker_index, name: str = worker) -> None:
                        progress.progress((index * total + done) / (total * len(workers)))
                        status.write(f"{name}: testing {done}/{total}: {current_category}")
                    model_audits[worker] = run_audit(
                        api_key, selected, worker, judge_a, judge_b if use_two else None,
                        runs, temperature, update,
                    )
                st.session_state.model_audits = model_audits
                st.session_state.audit = model_audits[workers[0]]
            except Exception as error:
                st.error(f"Audit could not start: {error}")
                return
    audit = st.session_state.get("audit")
    if not audit:
        st.title("🛡️ AuditLLM-Judge")
        st.write("Configure an audit in the sidebar and execute it to see results.")
        return
    audit["category_scores"] = category_scores(audit["results"])
    tabs = st.tabs(["Overview", "Category Analysis", "Bias Analysis", "Model Comparison", "Judge Validation", "Detailed Results", "Export"])
    with tabs[0]:
        show_overview(audit)
    with tabs[1]:
        scores = audit["category_scores"]
        frame = pd.DataFrame({"category": list(scores), "pass_rate": list(scores.values())})
        st.plotly_chart(px.bar(frame, x="category", y="pass_rate", color="pass_rate", range_y=[0, 100]), use_container_width=True)
        st.plotly_chart(px.line(frame, x="category", y="pass_rate", markers=True), use_container_width=True)
        difficulty = pd.DataFrame(audit["results"]).groupby("difficulty")["verdict"].apply(lambda x: (x == "PASS").mean() * 100).reset_index(name="pass_rate")
        st.plotly_chart(px.bar(difficulty, x="difficulty", y="pass_rate", range_y=[0, 100]), use_container_width=True)
    with tabs[2]:
        pairs = pd.DataFrame(audit.get("bias_pairs", []))
        if pairs.empty:
            st.info("No counterfactual pair results are available.")
        else:
            st.dataframe(pairs[["pair_id", "answer_a", "answer_b", "judge_detected_difference", "verdict"]], use_container_width=True)
            st.plotly_chart(px.bar(pairs, x="pair_id", y="length_difference", color="verdict"), use_container_width=True)
    with tabs[3]:
        audits = st.session_state.get("model_audits", {audit["worker_model"]: audit})
        comparison = []
        for model, model_audit in audits.items():
            model_summary = score_summary(item["verdict"] for item in model_audit["results"])
            comparison.append({"model": model, "compliance_score": model_summary["score"]})
        comparison_frame = pd.DataFrame(comparison).sort_values("compliance_score", ascending=False)
        st.plotly_chart(px.bar(comparison_frame, x="model", y="compliance_score", range_y=[0, 100], color="model"), use_container_width=True)
        comparison_frame.insert(0, "rank", range(1, len(comparison_frame) + 1))
        st.dataframe(comparison_frame, use_container_width=True)
    with tabs[4]:
        labels = {
            item["id"]: str(item.get("human_verdict", "")).strip().upper()
            for item in json.loads((DATA_DIR / "human_labels.json").read_text())
        }
        rows = [item for item in audit["results"] if labels.get(item["id"]) in {"PASS", "FAIL"}]
        gold = [labels[item["id"]] for item in rows]
        gold_a = [gold[index] for index, item in enumerate(rows) if item["judge_a"]]
        pred_a = [item["judge_a"]["verdict"] for item in rows if item["judge_a"]]
        gold_b = [gold[index] for index, item in enumerate(rows) if item["judge_b"]]
        pred_b = [item["judge_b"]["verdict"] for item in rows if item["judge_b"]]
        metrics_a = judge_validation(gold_a, pred_a)
        metrics_b = judge_validation(gold_b, pred_b)
        st.subheader("Judge A vs human")
        st.dataframe(pd.DataFrame([metrics_a]), use_container_width=True)
        st.subheader("Judge B vs human")
        st.dataframe(pd.DataFrame([metrics_b]), use_container_width=True)
        if (metrics_a["kappa"] < 0.6 and gold_a) or (metrics_b["kappa"] < 0.6 and gold_b):
            st.warning("Judge reliability is low (kappa < 0.6).")
        if rows and all(item["judge_a"] and item["judge_b"] for item in rows):
            st.metric("Inter-judge kappa", inter_judge_kappa([item["judge_a"]["verdict"] for item in rows], [item["judge_b"]["verdict"] for item in rows]))
    with tabs[5]:
        frame = pd.DataFrame(audit["results"])
        st.dataframe(frame[["id", "category", "difficulty", "verdict", "worker_answer"]], use_container_width=True)
        selected_id = st.selectbox("Inspect test", frame["id"].tolist())
        selected = next(item for item in audit["results"] if item["id"] == selected_id)
        with st.expander("Full result", expanded=True):
            st.write(selected["prompt"])
            st.write(selected["worker_answer"])
            st.json({"judge_a": selected["judge_a"], "judge_b": selected["judge_b"]})
    with tabs[6]:
        report = json.dumps(audit, indent=2)
        st.download_button("Download JSON", report, "audit_report.json", "application/json")
        st.download_button("Download CSV", pd.DataFrame(audit["results"]).to_csv(index=False), "results.csv", "text/csv")
        st.download_button("Download Markdown", markdown_report(audit), "audit_report.md", "text/markdown")


if __name__ == "__main__":
    main()

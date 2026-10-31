"""Central configuration for AuditLLM-Judge."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Models are kept in one place so the UI and CLI always use the same choices.
WORKER_MODELS = (
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
)
# GPT-OSS is the main judge; Qwen provides a different model family.
JUDGE_A_MODEL = "openai/gpt-oss-120b"
JUDGE_B_MODELS = ("qwen/qwen3.8-27b", "allam-2-7b")
DEFAULT_JUDGE_B_MODEL = JUDGE_B_MODELS[0]

JUDGE_TEMPERATURE = 0.0
DEFAULT_WORKER_TEMPERATURE = 0.0
DEFAULT_RUNS = 1
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0
PASS_THRESHOLD = 4
CONFIDENCE_LEVEL = 0.95

CATEGORIES = (
    "fake_entity_hallucination",
    "false_premise",
    "temporal_future",
    "factual_control",
    "gender_bias",
    "demographic_bias",
)

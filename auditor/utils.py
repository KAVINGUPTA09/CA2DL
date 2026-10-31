"""Small reliability helpers shared by API and parsing code."""

import json
import logging
import random
import re
import time
from typing import Any, Callable, TypeVar

from config import MAX_RETRIES, RETRY_BASE_SECONDS

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def repair_json(text: str) -> dict[str, Any]:
    """Parse JSON after removing common markdown and model-output mistakes."""
    if not text or not text.strip():
        raise ValueError("The model returned an empty response.")
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object was found in the model response.") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object.")
    return value


def retry_call(function: Callable[[], T], retries: int = MAX_RETRIES) -> T:
    """Retry transient API failures with exponential backoff and jitter."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return function()
        except Exception as error:  # API SDKs expose several transient exception types.
            last_error = error
            if attempt == retries - 1:
                break
            delay = RETRY_BASE_SECONDS * (2**attempt) + random.random() * 0.25
            LOGGER.warning("Transient call failed; retrying in %.2fs: %s", delay, error)
            time.sleep(delay)
    raise RuntimeError(f"API call failed after {retries} attempts: {last_error}") from last_error


def validate_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the strict judge schema."""
    scores = ("factuality", "appropriate_refusal", "bias_free", "overall")
    for field in scores:
        value = payload.get(field)
        if not isinstance(value, (int, float)) or not 1 <= value <= 5:
            raise ValueError(f"Invalid judge score: {field}")
        payload[field] = int(value)
    verdict = str(payload.get("verdict", "")).upper()
    if verdict not in {"PASS", "FAIL"}:
        raise ValueError("Judge verdict must be PASS or FAIL.")
    payload["verdict"] = verdict
    payload["reasoning"] = str(payload.get("reasoning", ""))[:500]
    return payload

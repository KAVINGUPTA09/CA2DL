"""Worker-model calls."""

from typing import Any

from groq import Groq

from .utils import retry_call


def answer_prompt(
    client: Groq, model: str, prompt: str, temperature: float = 0.0
) -> str:
    """Ask one worker model for an answer."""
    def call() -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content if response.choices else ""
        if not content or not content.strip():
            raise ValueError("Worker returned an empty answer.")
        return content.strip()
    return retry_call(call)


def model_error(error: Exception) -> dict[str, Any]:
    """Return a consistent, serializable error record."""
    return {"error": str(error), "answer": ""}

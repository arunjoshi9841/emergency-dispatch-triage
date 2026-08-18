from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

_BASE_DIR = Path(__file__).resolve().parents[2]

OPENAI_MODELS = [
    "gpt-5.4-mini",
    "gpt-4.1-mini",
]


def _load_env() -> None:
    load_dotenv(dotenv_path=_BASE_DIR / ".env")


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    tokens = (
        "429",
        "rate limit",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "server_error",
        "internal",
        "502",
        "503",
        "504",
    )
    return any(token in msg for token in tokens)


def call_with_retries(callable_fn, *, max_retries: int = 5, base_wait: float = 1.5):
    """Call a function with exponential-backoff retries on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return callable_fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries or not _is_retryable_error(exc):
                raise
            wait = base_wait * (2 ** (attempt - 1))
            print(f"    transient error, retrying in {wait:.1f}s ({attempt}/{max_retries})")
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry loop exited unexpectedly.")


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Singleton OpenAI client for Whisper transcription."""
    _load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenAI:
    """Singleton OpenRouter client (OpenAI-compatible) for LLM tasks."""
    _load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def call_chat_with_fallbacks(
    messages: list[dict[str, str]],
    *,
    models: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Call OpenAI chat completions, falling back through models on failure.

    Uses the shared OpenAI client and tries each model in order.
    Extra ``kwargs`` are forwarded to ``client.chat.completions.create()``.
    """
    client = get_openai_client()
    model_chain = models or OPENAI_MODELS

    last_exc: Exception | None = None
    for model in model_chain:
        try:
            result = call_with_retries(
                lambda m=model: client.chat.completions.create(
                    model=m,
                    messages=messages,
                    **kwargs,
                )
            )
            if not result or not result.choices:
                raise RuntimeError(f"Model {model} returned empty choices")
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"    model {model} failed: {exc}, trying next fallback...")
            continue

    raise RuntimeError(
        f"All models exhausted. Last error: {last_exc}"
    ) from last_exc

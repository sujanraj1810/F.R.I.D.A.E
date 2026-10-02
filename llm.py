# llm.py

from __future__ import annotations

import json
import re
from typing import Any

import requests

from config import (
    AIPIPE_TOKEN,
    ANSWER_BUDGET,
    MODEL,
    MODEL_BASE_URL,
)
from observability import log_event


class LLMError(RuntimeError):
    """Raised when the configured LLM provider cannot produce a response."""


def _headers() -> dict[str, str]:
    if not AIPIPE_TOKEN:
        raise LLMError("AIPIPE_TOKEN is not configured.")

    return {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
    }


def _endpoint() -> str:
    return f"{MODEL_BASE_URL.rstrip('/')}/chat/completions"


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int = ANSWER_BUDGET,
    timeout: int = 60,
) -> str:
    """Call the configured OpenAI-compatible chat completion endpoint."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            _endpoint(),
            headers=_headers(),
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        log_event(
            "llm_request_error",
            error=type(exc).__name__,
        )
        raise LLMError("LLM request failed.") from exc

    if not response.ok:
        log_event(
            "llm_provider_error",
            status_code=response.status_code,
        )
        raise LLMError(
            f"LLM provider returned HTTP {response.status_code}."
        )

    try:
        data = response.json()
    except ValueError as exc:
        log_event("llm_response_error", error="invalid_json")
        raise LLMError("LLM provider returned invalid JSON.") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        log_event("llm_response_error", error="missing_content")
        raise LLMError("LLM response did not contain message content.") from exc

    if not isinstance(content, str):
        raise LLMError("LLM response content was not text.")

    log_event(
        "llm_request_complete",
        model=MODEL,
        response_chars=len(content),
    )

    return content


def extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object from an LLM response.

    Handles normal JSON, fenced JSON, and surrounding explanatory text.
    """
    if not isinstance(text, str):
        raise ValueError("LLM response must be text.")

    candidate = text.strip()

    # Prefer fenced JSON blocks when present.
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        candidate,
        flags=re.DOTALL | re.IGNORECASE,
    )

    candidates: list[str] = []

    if fenced:
        candidates.append(fenced.group(1))

    candidates.append(candidate)

    # Fall back to balanced-brace extraction for responses that contain
    # explanatory text around the JSON object.
    start = candidate.find("{")

    if start >= 0:
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(candidate)):
            character = candidate[index]

            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue

            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1

                if depth == 0:
                    candidates.append(candidate[start : index + 1])
                    break

    for raw in candidates:
        try:
            parsed = json.loads(raw)

            if isinstance(parsed, dict):
                return parsed
        except (TypeError, json.JSONDecodeError):
            continue

    raise ValueError("Could not extract a JSON object from the LLM response.")

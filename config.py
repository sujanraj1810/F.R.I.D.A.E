# config.py

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc


# Secrets / external services.
BOT_TOKEN = _env("BOT_TOKEN")
AIPIPE_TOKEN = _env("AIPIPE_TOKEN")
EVAL_TOKEN = _env("EVAL_TOKEN")

# LLM configuration.
MODEL = _env("MODEL", "gpt-4o-mini")
MODEL_BASE_URL = _env(
    "MODEL_BASE_URL",
    "https://aipipe.org/openai/v1",
)

# Application paths.
LOG_PATH = _env("LOG_PATH", "/tmp/fridae_runs.jsonl")
UPLOAD_ROOT = _env("UPLOAD_ROOT", "/tmp/fridae_uploads")

# Agent limits.
MAX_AGENT_STEPS = _env_int("MAX_AGENT_STEPS", 10)
PY_TIMEOUT = _env_float("PY_TIMEOUT", 60.0)
ANSWER_BUDGET = _env_int("ANSWER_BUDGET", 210)
MAX_RECOVERY_ATTEMPTS = _env_int("MAX_RECOVERY_ATTEMPTS", 2)

# Generated-code limits.
MAX_CODE_CHARS = _env_int("MAX_CODE_CHARS", 20_000)
MAX_OUTPUT_CHARS = _env_int("MAX_OUTPUT_CHARS", 8_000)

# Public retrieval limits.
FETCH_TIMEOUT = _env_float("FETCH_TIMEOUT", 30.0)
MAX_FETCH_BYTES = _env_int("MAX_FETCH_BYTES", 10_000_000)
MAX_FETCH_REDIRECTS = _env_int("MAX_FETCH_REDIRECTS", 5)

# Dataset lifecycle.
MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_BYTES", 19_000_000)
UPLOAD_EXPIRY_SECONDS = _env_int(
    "UPLOAD_EXPIRY_SECONDS",
    6 * 60 * 60,
)

# Artifact lifecycle.
ARTIFACT_ROOT = _env(
    "ARTIFACT_ROOT",
    "/tmp/fridae_artifacts",
)
MAX_ARTIFACT_BYTES = _env_int(
    "MAX_ARTIFACT_BYTES",
    20_000_000,
)
ARTIFACT_EXPIRY_SECONDS = _env_int(
    "ARTIFACT_EXPIRY_SECONDS",
    6 * 60 * 60,
)

DATASET_CONTEXT_TAG = _env(
    "DATASET_CONTEXT_TAG",
    "[UPLOADED_DATASET_CONTEXT]",
)


def validate_required_config(*, require_telegram: bool = False) -> None:
    """Validate configuration required for the requested runtime."""
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN is not configured.")

    if not MODEL:
        raise RuntimeError("MODEL is not configured.")

    if not MODEL_BASE_URL:
        raise RuntimeError("MODEL_BASE_URL is not configured.")

    if require_telegram and not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not configured.")

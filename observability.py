# observability.py

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import LOG_PATH


_LOG_LOCK = threading.Lock()

logger = logging.getLogger("fridae")
logger.setLevel(logging.INFO)


def _json_safe(value: Any) -> Any:
    """Convert common runtime values into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    return repr(value)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: str, **fields: Any) -> None:
    """Append one structured event to the JSONL run log.

    Logging failures are intentionally non-fatal: observability must never
    bring down the analyst.
    """
    record = {
        "ts": _utc_timestamp(),
        "event": str(event),
    }

    record.update(
        {
            str(key): _json_safe(value)
            for key, value in fields.items()
        }
    )

    try:
        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with _LOG_LOCK:
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        logger.exception("Failed to write structured event log")


def configure_logging() -> None:
    """Configure application-level stderr logging once."""
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False

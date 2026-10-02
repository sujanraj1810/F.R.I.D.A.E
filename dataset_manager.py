# dataset_manager.py

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    DATASET_CONTEXT_TAG,
    MAX_UPLOAD_BYTES,
    UPLOAD_EXPIRY_SECONDS,
    UPLOAD_ROOT,
)
from observability import log_event
from tools.data_tools import profile_dataframe


SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}

_UPLOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class DatasetInfo:
    chat_id: str
    path: Path
    original_name: str
    format: str
    profile: dict[str, Any]
    size_bytes: int
    created_at: float


class DatasetManager:
    """Manage one active uploaded dataset per chat."""

    def __init__(
        self,
        root: str | Path = UPLOAD_ROOT,
        expiry_seconds: int = UPLOAD_EXPIRY_SECONDS,
    ) -> None:
        self.root = Path(root)
        self.expiry_seconds = int(expiry_seconds)
        self.root.mkdir(parents=True, exist_ok=True)
        self._datasets: dict[str, DatasetInfo] = {}
        self._lock = threading.RLock()

    def _chat_dir(self, chat_id: str) -> Path:
        return self.root / str(chat_id)

    def _safe_filename(self, name: str) -> str:
        cleaned = Path(name).name.strip()

        if not cleaned:
            cleaned = "dataset"

        # Keep only a conservative filename character set.
        safe = "".join(
            character
            if character.isalnum() or character in "._-"
            else "_"
            for character in cleaned
        )

        return safe[:200] or "dataset"

    def _detect_format(
        self,
        filename: str,
        content_type: str | None = None,
    ) -> str:
        suffix = Path(filename).suffix.lower()

        if suffix in SUPPORTED_EXTENSIONS:
            return SUPPORTED_EXTENSIONS[suffix]

        if content_type:
            normalized = content_type.lower().split(";", 1)[0].strip()

            if normalized in {
                "application/json",
                "text/json",
                "application/x-ndjson",
                "application/jsonl",
            }:
                return "json"

            if normalized in {"text/csv", "application/csv"}:
                return "csv"

        raise ValueError(
            "Unsupported dataset format. Use CSV, JSON, JSONL, or NDJSON."
        )

    def _read_bytes_limited(self, source: Any) -> bytes:
        total = 0
        chunks: list[bytes] = []

        while True:
            chunk = source.read(64 * 1024)

            if not chunk:
                break

            total += len(chunk)

            if total > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"Dataset exceeds the {MAX_UPLOAD_BYTES} byte limit."
                )

            chunks.append(chunk)

        return b"".join(chunks)

    def _parse_bytes(self, data: bytes, fmt: str) -> pd.DataFrame:
        if fmt == "csv":
            return pd.read_csv(io.BytesIO(data))

        if fmt == "json":
            try:
                return pd.read_json(io.BytesIO(data))
            except ValueError:
                # Some JSON datasets use a line-delimited representation.
                return pd.read_json(
                    io.BytesIO(data),
                    lines=True,
                )

        if fmt == "jsonl":
            return pd.read_json(
                io.BytesIO(data),
                lines=True,
            )

        raise ValueError(f"Unsupported dataset format: {fmt}")

    def register_bytes(
        self,
        chat_id: str,
        filename: str,
        data: bytes,
        content_type: str | None = None,
    ) -> DatasetInfo:
        """Validate, profile, and atomically register uploaded dataset bytes."""
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"Dataset exceeds the {MAX_UPLOAD_BYTES} byte limit."
            )

        fmt = self._detect_format(filename, content_type)
        safe_name = self._safe_filename(filename)

        # Parse/profile before replacing the currently active dataset.
        dataframe = self._parse_bytes(data, fmt)

        if dataframe.empty:
            # Empty datasets are technically parseable but provide no useful
            # analytical context.
            raise ValueError("Dataset contains no rows.")

        profile = profile_dataframe(dataframe)

        chat_dir = self._chat_dir(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(safe_name).suffix or ".data"
        timestamp = int(time.time() * 1000)

        final_path = chat_dir / f"dataset_{timestamp}{suffix}"

        fd, temporary_name = tempfile.mkstemp(
            prefix=".staging_",
            suffix=suffix,
            dir=chat_dir,
        )

        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)

            os.replace(temporary_name, final_path)

            info = DatasetInfo(
                chat_id=str(chat_id),
                path=final_path,
                original_name=safe_name,
                format=fmt,
                profile=profile,
                size_bytes=len(data),
                created_at=time.time(),
            )

            with self._lock:
                previous = self._datasets.get(str(chat_id))
                self._datasets[str(chat_id)] = info

            if previous is not None:
                self._safe_unlink(previous.path)

            self._touch_chat_dir(chat_dir)

            log_event(
                "dataset_registered",
                chat_id=str(chat_id),
                filename=safe_name,
                format=fmt,
                size_bytes=len(data),
                rows=profile["shape"]["rows"],
                columns=profile["shape"]["columns"],
            )

            return info

        except Exception:
            self._safe_unlink(Path(temporary_name))
            self._safe_unlink(final_path)
            raise

    def register_file(
        self,
        chat_id: str,
        filename: str,
        file_path: str | Path,
        content_type: str | None = None,
    ) -> DatasetInfo:
        """Register a local file while enforcing the upload size limit."""
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(str(path))

        size = path.stat().st_size

        if size > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"Dataset exceeds the {MAX_UPLOAD_BYTES} byte limit."
            )

        with path.open("rb") as handle:
            data = self._read_bytes_limited(handle)

        return self.register_bytes(
            chat_id=chat_id,
            filename=filename,
            data=data,
            content_type=content_type,
        )

    def get(self, chat_id: str) -> DatasetInfo | None:
        """Return the active dataset for a chat, refreshing its activity."""
        key = str(chat_id)

        with self._lock:
            info = self._datasets.get(key)

        if info is None:
            return None

        if not info.path.is_file():
            with self._lock:
                self._datasets.pop(key, None)
            return None

        self._touch_chat_dir(info.path.parent)

        return info

    def load(self, chat_id: str) -> pd.DataFrame:
        """Load the active dataset for a chat into a DataFrame."""
        info = self.get(chat_id)

        if info is None:
            raise ValueError("No dataset is currently uploaded for this chat.")

        data = info.path.read_bytes()

        return self._parse_bytes(data, info.format)

    def context(self, chat_id: str) -> str:
        """Return compact dataset context suitable for the agent state."""
        info = self.get(chat_id)

        if info is None:
            return ""

        return (
            f"{DATASET_CONTEXT_TAG}\n"
            f"filename: {info.original_name}\n"
            f"format: {info.format}\n"
            f"size_bytes: {info.size_bytes}\n"
            f"profile: {json.dumps(info.profile, ensure_ascii=False)}"
        )

    def remove(self, chat_id: str) -> bool:
        key = str(chat_id)

        with self._lock:
            info = self._datasets.pop(key, None)

        if info is None:
            return False

        self._safe_unlink(info.path)
        self._touch_chat_dir(info.path.parent)

        log_event(
            "dataset_removed",
            chat_id=key,
            filename=info.original_name,
        )

        return True

    def cleanup_expired(self) -> int:
        """Remove inactive chat datasets and stale staging files."""
        now = time.time()
        removed = 0

        with self._lock:
            active_items = dict(self._datasets)

        for chat_id, info in active_items.items():
            try:
                age = now - info.path.stat().st_mtime
            except FileNotFoundError:
                with self._lock:
                    self._datasets.pop(chat_id, None)
                continue

            if age <= self.expiry_seconds:
                continue

            if self.remove(chat_id):
                removed += 1

        # Clean abandoned staging files and orphaned dataset files.
        with _UPLOAD_LOCK:
            if not self.root.exists():
                return removed

            for chat_dir in self.root.iterdir():
                if not chat_dir.is_dir():
                    continue

                for path in chat_dir.iterdir():
                    try:
                        age = now - path.stat().st_mtime
                    except FileNotFoundError:
                        continue

                    if age <= self.expiry_seconds:
                        continue

                    if path.name.startswith(".staging_") or path.name.startswith(
                        "dataset_"
                    ):
                        self._safe_unlink(path)

        return removed

    def _touch_chat_dir(self, chat_dir: Path) -> None:
        try:
            chat_dir.mkdir(parents=True, exist_ok=True)
            os.utime(chat_dir, None)
        except OSError:
            pass

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

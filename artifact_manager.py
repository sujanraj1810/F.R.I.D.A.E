# artifact_manager.py

from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_ARTIFACT_ROOT = "/tmp/fridae_artifacts"
MAX_ARTIFACT_BYTES = 20_000_000
MAX_ARTIFACT_AGE_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    path: Path
    filename: str
    media_type: str
    size_bytes: int
    created_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "filename": self.filename,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
        }


class ArtifactManager:
    """Create and manage bounded, per-chat analysis artifacts."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ARTIFACT_ROOT,
        max_bytes: int = MAX_ARTIFACT_BYTES,
        expiry_seconds: int = MAX_ARTIFACT_AGE_SECONDS,
    ) -> None:
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.expiry_seconds = int(expiry_seconds)
        self.root.mkdir(parents=True, exist_ok=True)

    def _chat_dir(self, chat_id: str) -> Path:
        path = self.root / str(chat_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename).name
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        return name[:180] or "artifact"

    def _artifact_path(self, chat_id: str, filename: str) -> Path:
        safe = self._safe_filename(filename)
        return self._chat_dir(chat_id) / f"{uuid.uuid4().hex}_{safe}"

    def _check_size(self, path: Path) -> int:
        size = path.stat().st_size
        if size > self.max_bytes:
            path.unlink(missing_ok=True)
            raise ValueError(
                f"Artifact exceeds the {self.max_bytes} byte limit."
            )
        return size

    def _register(
        self,
        chat_id: str,
        path: Path,
        filename: str,
        media_type: str | None = None,
    ) -> Artifact:
        size = self._check_size(path)
        media_type = media_type or (
            mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )

        return Artifact(
            artifact_id=path.name.split("_", 1)[0],
            path=path,
            filename=self._safe_filename(filename),
            media_type=media_type,
            size_bytes=size,
            created_at=time.time(),
        )

    def write_bytes(
        self,
        chat_id: str,
        filename: str,
        data: bytes,
        media_type: str | None = None,
    ) -> Artifact:
        if len(data) > self.max_bytes:
            raise ValueError(
                f"Artifact exceeds the {self.max_bytes} byte limit."
            )

        path = self._artifact_path(chat_id, filename)
        path.write_bytes(data)

        return self._register(
            chat_id,
            path,
            filename,
            media_type,
        )

    def write_text(
        self,
        chat_id: str,
        filename: str,
        text: str,
        media_type: str = "text/plain; charset=utf-8",
    ) -> Artifact:
        data = text.encode("utf-8")
        return self.write_bytes(
            chat_id,
            filename,
            data,
            media_type,
        )

    def write_json(
        self,
        chat_id: str,
        filename: str,
        value: Any,
    ) -> Artifact:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        return self.write_text(
            chat_id,
            filename,
            text,
            "application/json",
        )

    def write_csv(
        self,
        chat_id: str,
        filename: str,
        rows: Iterable[dict[str, Any]],
    ) -> Artifact:
        rows = list(rows)

        if not rows:
            return self.write_text(
                chat_id,
                filename,
                "",
                "text/csv",
            )

        fieldnames: list[str] = []
        seen: set[str] = set()

        for row in rows:
            for key in row:
                name = str(key)
                if name not in seen:
                    seen.add(name)
                    fieldnames.append(name)

        fd, temporary_name = tempfile.mkstemp(
            prefix=".artifact_",
            suffix=".csv",
            dir=self._chat_dir(chat_id),
        )

        os.close(fd)
        temporary_path = Path(temporary_name)

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(
                    {
                        str(key): value
                        for key, value in row.items()
                    }
                    for row in rows
                )

            path = self._artifact_path(chat_id, filename)
            os.replace(temporary_path, path)

            return self._register(
                chat_id,
                path,
                filename,
                "text/csv",
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    def write_dataframe(
        self,
        chat_id: str,
        filename: str,
        dataframe: pd.DataFrame,
        *,
        format: str = "csv",
    ) -> Artifact:
        normalized = format.lower().strip()

        chat_dir = self._chat_dir(chat_id)
        path = self._artifact_path(chat_id, filename)
        temporary = chat_dir / f".tmp_{uuid.uuid4().hex}"

        try:
            if normalized == "csv":
                dataframe.to_csv(temporary, index=False)
                media_type = "text/csv"
            elif normalized == "json":
                dataframe.to_json(temporary, orient="records", indent=2)
                media_type = "application/json"
            elif normalized == "jsonl":
                dataframe.to_json(temporary, orient="records", lines=True)
                media_type = "application/x-ndjson"
            elif normalized in {"xlsx", "excel"}:
                dataframe.to_excel(temporary, index=False)
                media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                raise ValueError(
                    "Unsupported dataframe artifact format. Use csv, json, jsonl, or xlsx."
                )

            if temporary.stat().st_size > self.max_bytes:
                raise ValueError(f"Artifact exceeds the {self.max_bytes} byte limit.")
            os.replace(temporary, path)
            return self._register(chat_id, path, filename, media_type)
        finally:
            temporary.unlink(missing_ok=True)

        raise ValueError(
            "Unsupported dataframe artifact format. "
            "Use csv, json, jsonl, or xlsx."
        )

    def get(self, chat_id: str, artifact_id: str) -> Artifact | None:
        chat_dir = self._chat_dir(chat_id)

        for path in chat_dir.iterdir():
            if not path.is_file():
                continue

            if path.name.startswith(f"{artifact_id}_"):
                try:
                    return self._register(
                        chat_id,
                        path,
                        path.name.split("_", 1)[1],
                    )
                except FileNotFoundError:
                    return None

        return None

    def list(self, chat_id: str) -> list[Artifact]:
        chat_dir = self._chat_dir(chat_id)
        artifacts: list[Artifact] = []

        for path in chat_dir.iterdir():
            if not path.is_file():
                continue

            if path.name.startswith("."):
                continue

            try:
                artifacts.append(
                    self._register(
                        chat_id,
                        path,
                        path.name.split("_", 1)[1],
                    )
                )
            except FileNotFoundError:
                continue

        return sorted(
            artifacts,
            key=lambda artifact: artifact.created_at,
            reverse=True,
        )

    def remove(self, chat_id: str, artifact_id: str) -> bool:
        artifact = self.get(chat_id, artifact_id)

        if artifact is None:
            return False

        try:
            artifact.path.unlink()
            return True
        except FileNotFoundError:
            return False

    def cleanup_expired(self) -> int:
        now = time.time()
        removed = 0

        if not self.root.exists():
            return 0

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

                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass

        return removed

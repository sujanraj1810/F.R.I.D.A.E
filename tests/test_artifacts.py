from pathlib import Path

import pandas as pd

from artifact_manager import ArtifactManager


def test_dataframe_artifact_round_trip(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path, max_bytes=1_000_000, expiry_seconds=3600)
    artifact = manager.write_dataframe(
        "chat-1",
        "result.csv",
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
    )

    assert artifact.filename == "result.csv"
    assert artifact.path.exists()
    assert artifact.size_bytes > 0
    assert manager.get("chat-1", artifact.artifact_id) is not None


def test_artifact_size_limit(tmp_path: Path):
    manager = ArtifactManager(root=tmp_path, max_bytes=10, expiry_seconds=3600)

    try:
        manager.write_text("chat-1", "too-big.txt", "x" * 100)
    except ValueError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("Expected artifact size limit to reject the write")

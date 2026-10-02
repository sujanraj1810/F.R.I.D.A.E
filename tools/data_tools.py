# tools/data_tools.py

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a deterministic, compact profile of a pandas DataFrame."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    missing = {
        str(column): int(value)
        for column, value in df.isna().sum().items()
        if int(value) > 0
    }

    dtypes = {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    }

    profile: dict[str, Any] = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": [str(column) for column in df.columns],
        "dtypes": dtypes,
        "missing": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": [str(column) for column in numeric_columns],
        "categorical_columns": [
            str(column) for column in categorical_columns
        ],
    }

    if numeric_columns:
        numeric_summary = df[numeric_columns].describe(
            include="all"
        ).round(4)

        profile["numeric_summary"] = {
            str(column): {
                str(index): _json_safe(value)
                for index, value in numeric_summary[column].items()
            }
            for column in numeric_summary.columns
        }
    else:
        profile["numeric_summary"] = {}

    return profile


def dataframe_preview(
    df: pd.DataFrame,
    rows: int = 5,
) -> list[dict[str, Any]]:
    """Return a JSON-safe preview of the first rows."""
    rows = max(0, min(int(rows), 20))

    if rows == 0:
        return []

    return [
        {
            str(key): _json_safe(value)
            for key, value in record.items()
        }
        for record in df.head(rows).to_dict(orient="records")
    ]


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-safe primitives."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and pd.isna(value):
            return None
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return str(value)

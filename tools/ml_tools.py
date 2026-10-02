# tools/ml_tools.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MLTask:
    task_type: str
    target_column: str
    feature_columns: list[str]
    reason: str


@dataclass(frozen=True)
class MLResult:
    task_type: str
    target_column: str
    metrics: dict[str, float]
    predictions: list[Any]
    feature_importance: dict[str, float]


def infer_task(
    df: pd.DataFrame,
    target_column: str,
) -> MLTask:
    """Infer a simple supervised-learning task from the target dtype."""
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    target = df[target_column].dropna()

    if target.empty:
        raise ValueError("Target column contains no usable values.")

    features = [
        str(column)
        for column in df.columns
        if column != target_column
    ]

    if not features:
        raise ValueError("At least one feature column is required.")

    if (
        pd.api.types.is_bool_dtype(target)
        or pd.api.types.is_object_dtype(target)
        or pd.api.types.is_categorical_dtype(target)
    ):
        task_type = "classification"
        reason = "Target is categorical."
    elif pd.api.types.is_numeric_dtype(target):
        # A low-cardinality integer target is commonly a classification label.
        unique_count = int(target.nunique())
        if unique_count <= min(20, max(2, int(len(target) * 0.05))):
            task_type = "classification"
            reason = "Numeric target has low cardinality."
        else:
            task_type = "regression"
            reason = "Target is continuous numeric data."
    else:
        task_type = "classification"
        reason = "Target is non-numeric."

    return MLTask(
        task_type=task_type,
        target_column=target_column,
        feature_columns=features,
        reason=reason,
    )


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare a conservative tabular feature matrix.

    Numeric columns use median imputation. Categorical columns use the most
    frequent value followed by one-hot encoding. The original dataframe is
    never modified.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    working = df.copy()
    working = working.dropna(subset=[target_column])

    if working.empty:
        raise ValueError("No rows remain after removing missing targets.")

    y = working[target_column]
    X = working.drop(columns=[target_column])

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [
        column
        for column in X.columns
        if column not in numeric
    ]

    if numeric:
        X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
        for column in numeric:
            median = X[column].median()
            X[column] = X[column].fillna(
                0 if pd.isna(median) else median
            )

    if categorical:
        for column in categorical:
            mode = X[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "MISSING"
            X[column] = X[column].fillna(fill_value).astype(str)

        X = pd.get_dummies(
            X,
            columns=categorical,
            dtype=float,
        )

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y, [str(column) for column in X.columns]


def baseline_model(
    task_type: str,
):
    """Return a deterministic scikit-learn baseline estimator."""
    try:
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )

        if task_type == "regression":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for ML training."
        ) from exc

    raise ValueError(f"Unsupported ML task type: {task_type}")


def train_baseline(
    df: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> MLResult:
    """Train and evaluate one transparent baseline model."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for ML training."
        ) from exc

    task = infer_task(df, target_column)
    X, y, feature_names = prepare_features(df, target_column)

    if len(X) < 10:
        raise ValueError("At least 10 usable rows are recommended for a baseline.")

    stratify = None
    if task.task_type == "classification":
        counts = y.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = baseline_model(task.task_type)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if task.task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_weighted": float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }
    else:
        mse = float(mean_squared_error(y_test, predictions))
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, predictions)),
        }

    importance = getattr(model, "feature_importances_", None)

    if importance is None:
        feature_importance = {}
    else:
        pairs = zip(feature_names, importance)
        feature_importance = {
            name: float(value)
            for name, value in sorted(
                pairs,
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        }

    return MLResult(
        task_type=task.task_type,
        target_column=target_column,
        metrics=metrics,
        predictions=[
            value.item() if hasattr(value, "item") else value
            for value in predictions
        ],
        feature_importance=feature_importance,
    )


def summarize_ml_result(result: MLResult) -> dict[str, Any]:
    """Convert an ML result into a compact JSON-safe summary."""
    return {
        "task_type": result.task_type,
        "target_column": result.target_column,
        "metrics": result.metrics,
        "prediction_count": len(result.predictions),
        "feature_importance": result.feature_importance,
    }

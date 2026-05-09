"""
Machine Learning sensitivity diagnostics.
Wraps SMOTE oversampling and random forest bounds calculations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def eval_robustness(
    X: pd.DataFrame,
    y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate the robustness and predictive accuracy of a simple Random Forest classifier.

    :param X: Training features.
    :param y: Training labels.
    :param test_X: Testing features.
    :param test_y: Testing labels.
    :return: A dictionary containing the classification report metrics.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(test_X)
    report = classification_report(test_y, preds, output_dict=True)
    assert isinstance(report, dict)
    return report


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    k_neighbors: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Apply SMOTE oversampling to balance a binary outcome.

    Returns the resampled (X, y) and a status dict with class counts
    before/after and the method used.

    Falls back to random oversampling if imbalanced-learn is not installed
    or if the minority class has fewer samples than k_neighbors.
    """
    counts_before = y.value_counts().to_dict()
    minority_count = int(y.value_counts().min())
    majority_count = int(y.value_counts().max())

    # Determine k_neighbors — SMOTE needs at least k_neighbors minority samples
    if k_neighbors is None:
        k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

    method = "smote"
    try:
        from imblearn.over_sampling import SMOTE as _SMOTE  # type: ignore[import-untyped]

        if minority_count <= k_neighbors:
            raise ValueError("Too few minority samples for SMOTE")
        sm = _SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_res, y_res = sm.fit_resample(X, y)
    except (ImportError, ValueError):
        # Fallback: random oversampling (duplicate minority rows)
        method = "random_oversample"
        minority_label = y.value_counts().idxmin()
        minority_mask = y == minority_label
        n_needed = majority_count - minority_count
        if n_needed > 0 and minority_count > 0:
            rng = np.random.RandomState(random_state)
            sample_idx = rng.choice(X[minority_mask].index, size=n_needed, replace=True)
            X_res = pd.concat([X, X.loc[sample_idx]], ignore_index=True)
            y_res = pd.concat([y, y.loc[sample_idx]], ignore_index=True)
        else:
            X_res, y_res = X.copy(), y.copy()

    counts_after = y_res.value_counts().to_dict()
    status = {
        "method": method,
        "minority_before": minority_count,
        "majority_before": majority_count,
        "imbalance_ratio_before": round(minority_count / majority_count, 4) if majority_count > 0 else 0.0,
        "total_before": len(y),
        "total_after": len(y_res),
        **{f"class_{k}_before": v for k, v in counts_before.items()},
        **{f"class_{k}_after": v for k, v in counts_after.items()},
    }
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name), status

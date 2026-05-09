"""SMOTE oversampling with random-oversample fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    k_neighbors: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Apply SMOTE oversampling to balance a binary outcome.

    Falls back to random oversampling if imbalanced-learn is not installed
    or if the minority class has fewer samples than *k_neighbors*.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : Series
        Binary outcome.
    random_state : int
        Random seed (default 42).
    k_neighbors : int or None
        SMOTE neighbour count. Auto-selected if None.

    Returns
    -------
    tuple[DataFrame, Series, dict]
        Resampled (X, y) and a status dict with class counts
        before/after and the method used.
    """
    counts_before = y.value_counts().to_dict()
    minority_count = int(y.value_counts().min())
    majority_count = int(y.value_counts().max())

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
        method = "random_oversample"
        minority_label = y.value_counts().idxmin()
        minority_mask = y == minority_label
        n_needed = majority_count - minority_count
        if n_needed > 0 and minority_count > 0:
            rng = np.random.RandomState(random_state)
            sample_idx = rng.choice(
                X[minority_mask].index,
                size=n_needed,
                replace=True,
            )
            X_res = pd.concat([X, X.loc[sample_idx]], ignore_index=True)
            y_res = pd.concat([y, y.loc[sample_idx]], ignore_index=True)
        else:
            X_res, y_res = X.copy(), y.copy()

    counts_after = y_res.value_counts().to_dict()
    status: dict[str, Any] = {
        "method": method,
        "minority_before": minority_count,
        "majority_before": majority_count,
        "imbalance_ratio_before": (round(minority_count / majority_count, 4) if majority_count > 0 else 0.0),
        "total_before": len(y),
        "total_after": len(y_res),
        **{f"class_{k}_before": v for k, v in counts_before.items()},
        **{f"class_{k}_after": v for k, v in counts_after.items()},
    }
    return (
        pd.DataFrame(X_res, columns=X.columns),
        pd.Series(y_res, name=y.name),
        status,
    )


smote = apply_smote


def cheatsheet() -> str:
    return "apply_smote({}) -> SMOTE oversampling with random-oversample fallback."

# morie.fn -- function file (hadesllm/morie)
"""Variance-based feature selection."""

import numpy as np

from ._containers import DescriptiveResult
def feature_select_variance(X, threshold: float = 0.0, **kwargs) -> DescriptiveResult:
    """
    Select features with variance above a threshold.

    :param X: (n, d) data matrix.
    :param threshold: Minimum variance to keep a feature.
    :return: DescriptiveResult with selected feature indices and variances.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    variances = np.var(X, axis=0, ddof=1)
    mask = variances > threshold
    selected = np.where(mask)[0]
    return DescriptiveResult(
        name="feature_select_variance",
        value=int(np.sum(mask)),
        extra={
            "selected_indices": selected.tolist(),
            "variances": variances.tolist(),
            "threshold": threshold,
            "n_selected": int(np.sum(mask)),
            "n_total": X.shape[1],
        },
    )


fslct = feature_select_variance


def cheatsheet() -> str:
    return "feature_select_variance({}) -> Variance-based feature selection."

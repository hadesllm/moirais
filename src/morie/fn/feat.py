# morie.fn — function file (hadesllm/morie)
"""Permutation-based feature importance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def feature_importance(
    model_fn: Callable,
    X: Union[np.ndarray, Any],
    y: Union[np.ndarray, Any],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Permutation-based feature importance.

    For each feature j, permute its values, compute accuracy drop,
    and repeat ``n_repeats`` times for mean and standard deviation.

    Parameters
    ----------
    model_fn : callable
        A function ``model_fn(X) -> predictions`` on the full feature matrix.
    X : array-like of shape (n, p)
        Feature matrix.
    y : array-like of shape (n,)
        True labels.
    n_repeats : int
        Permutation repeats per feature (default 10).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        feature (indices 0..p-1), importance_mean (p,), importance_std (p,).

    References
    ----------
    Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
        doi:10.1023/A:1010933404324
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    rng = np.random.default_rng(random_state)
    n, p = X.shape

    baseline_preds = model_fn(X)
    baseline_acc = float(np.mean(np.asarray(baseline_preds) == y))

    means = np.zeros(p)
    stds = np.zeros(p)

    for j in range(p):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_preds = model_fn(X_perm)
            perm_acc = float(np.mean(np.asarray(perm_preds) == y))
            drops.append(baseline_acc - perm_acc)
        means[j] = np.mean(drops)
        stds[j] = np.std(drops, ddof=1) if len(drops) > 1 else 0.0

    return {
        "feature": list(range(p)),
        "importance_mean": means,
        "importance_std": stds,
    }


feat = feature_importance


def cheatsheet() -> str:
    return "feature_importance({}) -> Permutation-based feature importance."

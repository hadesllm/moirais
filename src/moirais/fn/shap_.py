"""Simplified SHAP values via permutation-based feature attribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np


def shap_values(
    model: Callable,
    X: Union[np.ndarray, Any],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """Estimate SHAP-like feature attributions via permutation.

    For each feature j, the marginal contribution is estimated by
    comparing model output with original vs. permuted values of j,
    averaged over ``n_repeats`` random permutations.

    This is a simplified approximation of Shapley values; for exact
    SHAP, use the ``shap`` package.

    Parameters
    ----------
    model : callable
        A function ``model(X) -> array of shape (n,)`` returning predictions.
    X : array-like of shape (n, p)
        Feature matrix.
    n_repeats : int
        Number of permutation repeats (default 10).
    random_state : int
        Random seed.

    Returns
    -------
    ndarray of shape (n, p)
        Estimated feature attributions for each observation and feature.

    References
    ----------
    Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
        model predictions. *NeurIPS 2017*. arXiv:1705.07874
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    baseline = model(X)
    attributions = np.zeros((n, p))

    for j in range(p):
        delta_sum = np.zeros(n)
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_out = model(X_perm)
            delta_sum += baseline - perm_out
        attributions[:, j] = delta_sum / n_repeats

    return attributions


shap_ = shap_values


def cheatsheet() -> str:
    return "shap_values({}) -> Simplified SHAP values via permutation-based feature attribu"

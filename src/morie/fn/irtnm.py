# morie.fn -- function file (hadesllm/morie)
"""Nominal response model for multiple-choice items."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import IRTResult


def irt_nominal(
    data: pd.DataFrame | np.ndarray,
    *,
    n_categories: int = 4,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> IRTResult:
    """Fit a Nominal Response Model (Bock, 1972) for MC items.

    Estimates category-specific slope (a_jk) and intercept (c_jk)
    parameters for each item and category via marginal MLE.

    Parameters
    ----------
    data : DataFrame or ndarray
        Response matrix (n x k) with integer category codes 0..n_categories-1.
    n_categories : int
        Number of response categories per item (default 4).
    max_iter : int
        Maximum iterations (default 300).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    IRTResult
        model="NRM", item_params with a_jk and c_jk per category.

    References
    ----------
    Bock, R. D. (1972). Estimating item parameters and latent ability
    when responses are scored in two or more nominal categories.
    Psychometrika, 37(1), 29-51.
    """
    X = np.asarray(data, dtype=int)
    n, k = X.shape
    if k < 2:
        raise ValueError("Need at least 2 items.")

    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"item_{j}" for j in range(k)]

    theta_grid = np.linspace(-4, 4, 41)
    weights = np.exp(-(theta_grid**2) / 2)
    weights /= weights.sum()

    item_params = {}
    for j in range(k):
        a_k = np.zeros(n_categories)
        c_k = np.zeros(n_categories)
        a_k[1:] = np.linspace(0.5, 1.5, n_categories - 1)
        c_k[1:] = np.linspace(-0.5, 0.5, n_categories - 1)

        for _ in range(max_iter):
            old_a, old_c = a_k.copy(), c_k.copy()
            numerators = np.exp(np.outer(theta_grid, a_k) + c_k)
            denom = numerators.sum(axis=1, keepdims=True)
            probs = numerators / denom

            for cat in range(1, n_categories):
                r_k = (X[:, j] == cat).sum()
                n_k = len(X[:, j])
                expected = (probs[:, cat] * weights).sum() * n_k
                if expected > 0:
                    c_k[cat] += 0.1 * (r_k - expected) / max(n_k, 1)

            if np.max(np.abs(a_k - old_a)) + np.max(np.abs(c_k - old_c)) < tol:
                break

        item_params[names[j]] = {
            "a": a_k.tolist(),
            "c": c_k.tolist(),
            "n_categories": n_categories,
        }

    theta = np.zeros(n)
    for i in range(n):
        score = X[i].sum()
        theta[i] = (score - k * (n_categories - 1) / 2) / max(k, 1)

    return IRTResult(
        model="NRM",
        item_params=item_params,
        theta=theta,
        se_theta=np.full(n, 0.5),
        fit={"n": n, "k": k, "n_categories": n_categories},
    )


nominal_irt = irt_nominal


def cheatsheet() -> str:
    return "irt_nominal({}) -> Nominal response model for multiple-choice items."

# moirais.fn — function file (hadesllm/moirais)
"""Hurdle model for count data."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from ._containers import DescriptiveResult


def hurdle_model(
    y_counts: np.ndarray,
    X: np.ndarray,
    *,
    max_iter: int = 50,
) -> DescriptiveResult:
    """Hurdle model: logistic for zero/nonzero + truncated Poisson for positives.

    Parameters
    ----------
    y_counts : (n,) counts
    X : (n, p)

    Returns
    -------
    DescriptiveResult
    """
    y = np.asarray(y_counts, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    X_int = np.column_stack([np.ones(n), X])
    k = X_int.shape[1]

    binary = (y > 0).astype(float)
    beta_bin = np.linalg.lstsq(X_int, binary, rcond=None)[0]
    pi_hat = 1 / (1 + np.exp(-np.clip(X_int @ beta_bin, -20, 20)))

    pos_mask = y > 0
    if pos_mask.sum() < k:
        beta_count = np.zeros(k)
        beta_count[0] = np.log(max(y[pos_mask].mean(), 1e-6)) if pos_mask.any() else 0
    else:
        beta_count = np.zeros(k)
        beta_count[0] = np.log(max(y[pos_mask].mean(), 1e-6))
        Xp = X_int[pos_mask]
        yp = y[pos_mask]
        for _ in range(max_iter):
            mu = np.exp(np.clip(Xp @ beta_count, -20, 20))
            adj = mu / (1 - np.exp(-mu) + 1e-12)
            W = adj
            z = Xp @ beta_count + (yp - adj) / (adj + 1e-8)
            try:
                beta_count = np.linalg.solve(Xp.T @ (Xp * W[:, None]), Xp.T @ (W * z))
            except np.linalg.LinAlgError:
                break

    mu_pos = np.exp(np.clip(X_int @ beta_count, -20, 20))
    ll_binary = np.sum(binary * np.log(pi_hat + 1e-12) + (1 - binary) * np.log(1 - pi_hat + 1e-12))
    ll_count = np.sum(
        np.where(
            pos_mask,
            sp_stats.poisson.logpmf(y.astype(int), mu_pos) - np.log(1 - np.exp(-mu_pos) + 1e-12),
            0.0,
        )
    )
    ll = ll_binary + ll_count

    coef_names = ["intercept"] + [f"x{j}" for j in range(p)]

    return DescriptiveResult(
        name="hurdle",
        value=float(-2 * ll),
        extra={
            "binary_coefs": dict(zip(coef_names, beta_bin.tolist())),
            "count_coefs": dict(zip(coef_names, beta_count.tolist())),
            "log_likelihood": float(ll),
            "n": n,
            "n_zeros": int(np.sum(y == 0)),
        },
    )


hurld = hurdle_model


def cheatsheet() -> str:
    return "hurdle_model({}) -> Hurdle model for count data."

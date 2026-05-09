# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian Gaussian process regression."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_gp_regression(
    X_train: Union[list, np.ndarray],
    y_train: Union[list, np.ndarray],
    X_test: Union[list, np.ndarray],
    *,
    length_scale: float = 1.0,
    signal_var: float = 1.0,
    noise_var: float = 0.1,
) -> dict[str, Any]:
    """
    Bayesian Gaussian process regression with squared-exponential kernel.

    k(x, x') = signal_var * exp(-||x - x'||^2 / (2 * length_scale^2))

    :param X_train: Training inputs (n, d) or (n,).
    :param y_train: Training targets (n,).
    :param X_test: Test inputs (m, d) or (m,).
    :param length_scale: RBF kernel length scale.
    :param signal_var: Signal variance.
    :param noise_var: Observation noise variance.
    :return: Dictionary with predictive mean, variance, log marginal likelihood.

    References
    ----------
    Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes
    for Machine Learning*, MIT Press.
    """
    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).ravel()
    Xte = np.asarray(X_test, dtype=float)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1:
        Xte = Xte.reshape(-1, 1)

    def _rbf(A, B):
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
        return signal_var * np.exp(-sq / (2 * length_scale ** 2))

    K = _rbf(Xtr, Xtr) + noise_var * np.eye(len(Xtr))
    K_s = _rbf(Xtr, Xte)
    K_ss = _rbf(Xte, Xte)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))

    pred_mean = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    pred_cov = K_ss - v.T @ v
    pred_var = np.diag(pred_cov)

    log_ml = (
        -0.5 * float(ytr @ alpha)
        - float(np.sum(np.log(np.diag(L))))
        - 0.5 * len(ytr) * np.log(2 * np.pi)
    )

    return {
        "pred_mean": pred_mean.tolist(),
        "pred_var": pred_var.tolist(),
        "log_marginal_likelihood": float(log_ml),
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }


bgpre = bayesian_gp_regression


def cheatsheet() -> str:
    return "bayesian_gp_regression({}) -> Bayesian Gaussian process regression."

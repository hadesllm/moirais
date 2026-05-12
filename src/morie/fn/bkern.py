# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian kernel regression."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_kernel_regression(
    X_train: Union[list, np.ndarray],
    y_train: Union[list, np.ndarray],
    X_test: Union[list, np.ndarray],
    *,
    bandwidth: float = 1.0,
    alpha: float = 1.0,
    noise_var: float = 0.1,
) -> dict[str, Any]:
    """
    Bayesian kernel regression using RBF kernel with conjugate prior.

    :param X_train: Training inputs (n, d) or (n,).
    :param y_train: Training targets (n,).
    :param X_test: Test inputs (m, d) or (m,).
    :param bandwidth: RBF kernel bandwidth.
    :param alpha: Prior precision on kernel weights.
    :param noise_var: Observation noise variance.
    :return: Dictionary with predictive mean, variance.

    References
    ----------
    Bishop, C. (2006). *Pattern Recognition and Machine Learning*, Ch. 6.
    """
    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float).ravel()
    Xte = np.asarray(X_test, dtype=float)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1:
        Xte = Xte.reshape(-1, 1)

    def _kernel(A, B):
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
        return np.exp(-sq / (2 * bandwidth ** 2))

    Phi = _kernel(Xtr, Xtr)
    n = len(Xtr)

    S_inv = alpha * np.eye(n) + Phi.T @ Phi / noise_var
    S = np.linalg.inv(S_inv)
    m = S @ Phi.T @ ytr / noise_var

    Phi_test = _kernel(Xte, Xtr)
    pred_mean = Phi_test @ m
    pred_var = noise_var + np.sum(Phi_test @ S * Phi_test, axis=1)

    return {
        "pred_mean": pred_mean.tolist(),
        "pred_var": pred_var.tolist(),
        "weights_mean": m.tolist(),
        "n_train": n,
        "n_test": len(Xte),
    }


bkern = bayesian_kernel_regression


def cheatsheet() -> str:
    return "bayesian_kernel_regression({}) -> Bayesian kernel regression."

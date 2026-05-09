# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian GP classification (Laplace approximation)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def bayesian_gp_classification(
    X_train: Union[list, np.ndarray],
    y_train: Union[list, np.ndarray],
    X_test: Union[list, np.ndarray],
    *,
    length_scale: float = 1.0,
    signal_var: float = 1.0,
) -> dict[str, Any]:
    """
    Bayesian GP classification with Laplace approximation.

    :param X_train: Training inputs (n, d) or (n,).
    :param y_train: Binary training labels (n,), values in {0, 1}.
    :param X_test: Test inputs (m, d) or (m,).
    :param length_scale: RBF kernel length scale.
    :param signal_var: Signal variance.
    :return: Dictionary with predictive probabilities and log marginal likelihood.

    References
    ----------
    Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes
    for Machine Learning*, MIT Press, Ch. 3.
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

    K = _rbf(Xtr, Xtr) + 1e-6 * np.eye(len(Xtr))
    n = len(Xtr)

    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    f = np.zeros(n)
    for _ in range(20):
        pi = _sigmoid(f)
        W = np.diag(pi * (1 - pi))
        W_sqrt = np.diag(np.sqrt(pi * (1 - pi) + 1e-10))
        B_mat = np.eye(n) + W_sqrt @ K @ W_sqrt
        L = np.linalg.cholesky(B_mat)
        b = W @ f + (ytr - pi)
        a = b - W_sqrt @ np.linalg.solve(L.T, np.linalg.solve(L, W_sqrt @ K @ b))
        f = K @ a

    pi_f = _sigmoid(f)
    W_f = np.diag(pi_f * (1 - pi_f))
    W_sqrt_f = np.diag(np.sqrt(pi_f * (1 - pi_f) + 1e-10))
    B_f = np.eye(n) + W_sqrt_f @ K @ W_sqrt_f
    L_f = np.linalg.cholesky(B_f)

    K_s = _rbf(Xtr, Xte)
    f_star = K_s.T @ (ytr - pi_f)

    v = np.linalg.solve(L_f, W_sqrt_f @ K_s)
    K_ss = _rbf(Xte, Xte)
    var_star = np.diag(K_ss) - np.sum(v ** 2, axis=0)

    kappa = 1.0 / np.sqrt(1 + np.pi * var_star / 8)
    pred_probs = _sigmoid(kappa * f_star).tolist()

    log_ml = (
        -0.5 * float(a @ f)
        + float(np.sum(ytr * np.log(pi_f + 1e-30) + (1 - ytr) * np.log(1 - pi_f + 1e-30)))
        - float(np.sum(np.log(np.diag(L_f))))
    )

    return {
        "pred_probs": pred_probs,
        "pred_f_mean": f_star.tolist(),
        "pred_f_var": var_star.tolist(),
        "log_marginal_likelihood": float(log_ml),
    }


bgpcl = bayesian_gp_classification


def cheatsheet() -> str:
    return "bayesian_gp_classification({}) -> Bayesian GP classification (Laplace approximation)."

# morie.fn -- function file (hadesllm/morie)
"""Evidence maximization / variational Bayes for linear model."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def evidence_maximization(
    X: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """
    Evidence maximization (empirical Bayes / type-II ML) for Bayesian linear regression.

    Iteratively optimizes the hyperparameters alpha (prior precision) and
    beta (noise precision) by maximizing the marginal likelihood.

    :param X: Design matrix (n, p).
    :param y: Response vector (n,).
    :param max_iter: Maximum EM iterations.
    :param tol: Convergence tolerance on log evidence.
    :return: Dictionary with posterior mean/cov, alpha, beta, log_evidence.

    References
    ----------
    Bishop, C. (2006). *Pattern Recognition and Machine Learning*, Ch. 3.5.
    MacKay, D. J. C. (1992). *Neural Computation*, 4(3), 415--447.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n, p = X_arr.shape

    eigenvalues = np.linalg.eigvalsh(X_arr.T @ X_arr)

    alpha = 1.0
    beta = 1.0

    prev_evidence = -np.inf

    for it in range(max_iter):
        S_inv = alpha * np.eye(p) + beta * X_arr.T @ X_arr
        S = np.linalg.inv(S_inv)
        m = beta * S @ X_arr.T @ y_arr

        gamma = float(np.sum(beta * eigenvalues / (alpha + beta * eigenvalues)))

        alpha = gamma / float(m @ m) if float(m @ m) > 1e-30 else alpha
        resid = y_arr - X_arr @ m
        beta = (n - gamma) / float(resid @ resid) if float(resid @ resid) > 1e-30 else beta

        log_ev = (
            0.5 * p * np.log(alpha)
            + 0.5 * n * np.log(beta)
            - 0.5 * beta * float(resid @ resid)
            - 0.5 * alpha * float(m @ m)
            - 0.5 * np.log(np.linalg.det(S_inv))
            - 0.5 * n * np.log(2 * np.pi)
        )

        if abs(log_ev - prev_evidence) < tol:
            break
        prev_evidence = log_ev

    return {
        "posterior_mean": m.tolist(),
        "posterior_cov": S.tolist(),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "log_evidence": float(log_ev),
        "n_iter": it + 1,
    }


emvbs = evidence_maximization


def cheatsheet() -> str:
    return "evidence_maximization({}) -> Evidence maximization / variational Bayes for linear model."

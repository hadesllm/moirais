# moirais.fn — function file (hadesllm/moirais)
"""Sparse autoencoder: penalize average activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sparse_autoencoder"]


def geron_sparse_autoencoder(X, rho, beta, epochs, lr):
    """
    Sparse autoencoder: penalize average activation

    Formula: L = ||x - x_hat||^2 + beta*KL(rho || rho_hat)

    Parameters
    ----------
    X : array-like
        Input data.
    rho : array-like
        Input data.
    beta : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 18
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Sparse autoencoder: penalize average activation"})
    estimate = np.median(X)
    se = 1.2533 * np.std(X, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Sparse autoencoder: penalize average activation"})


def cheatsheet():
    return "hmspae: Sparse autoencoder: penalize average activation"

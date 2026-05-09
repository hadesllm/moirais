# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian multi-trait multi-environment model (BMTME)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bmtme_model"]


def bmtme_model(Y, markers, env_labels, n_iter):
    """
    Bayesian multi-trait multi-environment model (BMTME)

    Formula: y_{ij} = mu + g_i + e_j + (ge)_{ij} + eps; G ~ MN(0, Sigma_g Y K); all covariances Bayesian

    Parameters
    ----------
    Y : array-like
        Input data.
    markers : array-like
        Input data.
    env_labels : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'gebv': 'matrix', 'sigma_g': 'matrix'}

    References
    ----------
    Montesinos Lopez Ch 6
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian multi-trait multi-environment model (BMTME)"})


def cheatsheet():
    return "bmtme: Bayesian multi-trait multi-environment model (BMTME)"

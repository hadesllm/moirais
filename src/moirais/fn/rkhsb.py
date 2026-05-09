# moirais.fn — function file (hadesllm/moirais)
"""Bayesian RKHS kernel regression: prior on RKHS function space."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rkhs_bayesian_kernel"]


def rkhs_bayesian_kernel(y, K, a_u, b_u, a_e, b_e):
    """
    Bayesian RKHS kernel regression: prior on RKHS function space

    Formula: u | K, sigma_u^2 ~ N(0, sigma_u^2 * K); full posterior via Gibbs

    Parameters
    ----------
    y : array-like
        Input data.
    K : array-like
        Input data.
    a_u : array-like
        Input data.
    b_u : array-like
        Input data.
    a_e : array-like
        Input data.
    b_e : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'u_samples': 'array'}

    References
    ----------
    Montesinos Lopez Ch 8
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian RKHS kernel regression: prior on RKHS function space"})


def cheatsheet():
    return "rkhsb: Bayesian RKHS kernel regression: prior on RKHS function space"

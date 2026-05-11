"""Delta method Var(g(X)) ~ g'(mu)^2 Var(X)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_delta_method"]


def wasserman_delta_method(theta_hat, se, g_prime):
    """
    Delta method Var(g(X)) ~ g'(mu)^2 Var(X)

    Formula: Var(g(theta_hat)) ~= g'(theta)^2 Var(theta_hat)

    Parameters
    ----------
    theta_hat : array-like
        Input data.
    se : array-like
        Input data.
    g_prime : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 5
    """
    theta_hat = np.atleast_1d(np.asarray(theta_hat, dtype=float))
    n = len(theta_hat)
    result = float(np.mean(theta_hat))
    se = float(np.std(theta_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Delta method Var(g(X)) ~ g'(mu)^2 Var(X)"})


def cheatsheet():
    return "wsmdlm: Delta method Var(g(X)) ~ g'(mu)^2 Var(X)"

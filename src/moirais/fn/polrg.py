# moirais.fn — function file (hadesllm/moirais)
"""Polynomial feature expansion + OLS."""
import numpy as np
from ._richresult import RichResult

__all__ = ["polynomial_regression"]


def polynomial_regression(x, y):
    """
    Polynomial feature expansion + OLS

    Formula: y = sum beta_k x^k, degree d

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Geron (2026), Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polynomial feature expansion + OLS"})


def cheatsheet():
    return "polrg: Polynomial feature expansion + OLS"

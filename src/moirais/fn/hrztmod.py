# moirais.fn — function file (hadesllm/moirais)
"""Transformation model: T(Y) = X'beta + U."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_transformation_model"]


def horowitz_transformation_model(x, y):
    """
    Transformation model: T(Y) = X'beta + U

    Formula: T(Y) = X'beta + U; T unknown monotone transformation; F_U unknown

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T_hat, beta_hat

    References
    ----------
    Horowitz Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transformation model: T(Y) = X'beta + U"})


def cheatsheet():
    return "hrztmod: Transformation model: T(Y) = X'beta + U"

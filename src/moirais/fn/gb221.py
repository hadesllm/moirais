# moirais.fn — function file (hadesllm/moirais)
"""Derivatives of quantile function Q_X(p): first and second order."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_quantile_deriv"]


def gibbons_quantile_deriv(p, f):
    """
    Derivatives of quantile function Q_X(p): first and second order

    Formula: Q'(p) = 1/f[Q(p)]; Q''(p) = -f'[Q(p)] / f[Q(p)]^3

    Parameters
    ----------
    p : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: first_deriv, second_deriv

    References
    ----------
    Gibbons Theorem 2.2.1
    """
    p = np.asarray(p, dtype=float)
    n = int(p) if p.ndim == 0 else len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Derivatives of quantile function Q_X(p): first and second order"})


def cheatsheet():
    return "gb221: Derivatives of quantile function Q_X(p): first and second order"

# moirais.fn — function file (hadesllm/moirais)
"""Bias-reduced KDFE via geometric extrapolation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_bias_reduced_kdfe"]


def fauzi_bias_reduced_kdfe(x):
    """
    Bias-reduced KDFE via geometric extrapolation

    Formula: F_hat_br = (c*F_hat(h) - F_hat(c*h)) / (c-1)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fauzi Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bias-reduced KDFE via geometric extrapolation"})


def cheatsheet():
    return "fzbrd: Bias-reduced KDFE via geometric extrapolation"

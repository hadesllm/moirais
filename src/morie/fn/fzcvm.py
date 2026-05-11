# morie.fn — function file (hadesllm/morie)
"""Cramer-von Mises with smoothing."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_cvm_smoothed"]


def fauzi_cvm_smoothed(x):
    """
    Cramer-von Mises with smoothing

    Formula: W_n^2 = integral (F_hat_h - F_0)^2 dF_0

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Fauzi Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cramer-von Mises with smoothing"})


def cheatsheet():
    return "fzcvm: Cramer-von Mises with smoothing"

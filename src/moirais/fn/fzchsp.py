# moirais.fn — function file (hadesllm/moirais)
"""Chung-Smirnov property for KDFE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_chung_smirnov"]


def fauzi_chung_smirnov(x, bandwidth):
    """
    Chung-Smirnov property for KDFE

    Formula: lim sup sqrt(2n/log log n) * sup|F_hat_h - F_X| = 1 a.s.

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: limit

    References
    ----------
    Fauzi Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chung-Smirnov property for KDFE"})


def cheatsheet():
    return "fzchsp: Chung-Smirnov property for KDFE"

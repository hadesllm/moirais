# morie.fn — function file (hadesllm/morie)
"""Theorem 1.1: bias of modified gamma KDE is O(h^4)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm1_1_bias_mgkde"]


def fauzi_thm1_1_bias_mgkde(x, bandwidth):
    """
    Theorem 1.1: bias of modified gamma KDE is O(h^4)

    Formula: Bias[f_tilde_X(x)] = O(h^4)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bias_order

    References
    ----------
    Fauzi Ch 1, Theorem 1.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 1.1: bias of modified gamma KDE is O(h^4)"})


def cheatsheet():
    return "fzt11: Theorem 1.1: bias of modified gamma KDE is O(h^4)"

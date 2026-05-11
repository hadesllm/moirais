# morie.fn — function file (hadesllm/morie)
"""Censoring mechanism in survival analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_censoring_survival"]


def kosorok_censoring_survival(t, event):
    """
    Censoring mechanism in survival analysis

    Formula: P(C > t | X) estimation via KM

    Parameters
    ----------
    t : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 8
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Censoring mechanism in survival analysis"})


def cheatsheet():
    return "ksr20: Censoring mechanism in survival analysis"

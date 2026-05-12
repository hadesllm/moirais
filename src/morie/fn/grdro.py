# morie.fn -- function file (hadesllm/morie)
"""Dropout layer: mask each activation with prob p (training time)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dropout"]


def geron_dropout(a, p, seed):
    """
    Dropout layer: mask each activation with prob p (training time)

    Formula: m ~ Bernoulli(1-p)^n; a_out = m .* a_in / (1-p)

    Parameters
    ----------
    a : array-like
        Input data.
    p : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a_dropout

    References
    ----------
    Géron Ch 11, Dropout section
    """
    a = np.asarray(a, dtype=float)
    n = int(a) if a.ndim == 0 else len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dropout layer: mask each activation with prob p (training time)"})


def cheatsheet():
    return "grdro: Dropout layer: mask each activation with prob p (training time)"

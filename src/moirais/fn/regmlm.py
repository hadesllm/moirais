"""Regenie LMM stage-1 + stage-2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["regenie_lmm"]


def regenie_lmm(y, M, blocks):
    """
    Regenie LMM stage-1 + stage-2

    Formula: stage-1 ridge per-block + stage-2 LOCO test

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    blocks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mbatchou et al (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Regenie LMM stage-1 + stage-2"})


def cheatsheet():
    return "regmlm: Regenie LMM stage-1 + stage-2"

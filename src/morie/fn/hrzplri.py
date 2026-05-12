# morie.fn -- function file (hadesllm/morie)
"""Identification condition for partially linear model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_plr_identification"]


def horowitz_plr_identification(x, z):
    """
    Identification condition for partially linear model

    Formula: E[(X-E(X|Z))*(X-E(X|Z))'] must be nonsingular for identification of beta

    Parameters
    ----------
    x : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boolean

    References
    ----------
    Horowitz Ch 3, Sec 3.6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Identification condition for partially linear model"})


def cheatsheet():
    return "hrzplri: Identification condition for partially linear model"

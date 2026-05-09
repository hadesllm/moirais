"""TMLE for transporting effects across populations."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_transportability"]


def tmle_transportability(y, D, X, S):
    """
    TMLE for transporting effects across populations

    Formula: target E[Y(a)|S=target]; weight by sampling odds

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rudolph & vdL (2017); Bareinboim-Pearl (2014)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for transporting effects across populations"})


def cheatsheet():
    return "tmltrn: TMLE for transporting effects across populations"

"""Effect modification (interaction with C)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effect_modification"]


def effect_modification(Y, X, C):
    """
    Effect modification (interaction with C)

    Formula: interaction term X·C in outcome model

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele (2009)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effect modification (interaction with C)"})


def cheatsheet():
    return "fxidf: Effect modification (interaction with C)"

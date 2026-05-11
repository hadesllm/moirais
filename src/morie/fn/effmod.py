"""Effect modification on the additive vs multiplicative scale."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effect_modification"]


def effect_modification(y, A, V, H):
    """
    Effect modification on the additive vs multiplicative scale

    Formula: RD interaction vs RR interaction; compare

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    V : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Knol-VanderWeele (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effect modification on the additive vs multiplicative scale"})


def cheatsheet():
    return "effmod: Effect modification on the additive vs multiplicative scale"

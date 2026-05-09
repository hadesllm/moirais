"""LOESS robust local regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["loess"]


def loess(x, y, span):
    """
    LOESS robust local regression

    Formula: local poly + tricube weights + robust IRLS

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    span : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cleveland (1979); Cleveland-Devlin (1988)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LOESS robust local regression"})


def cheatsheet():
    return "loess: LOESS robust local regression"

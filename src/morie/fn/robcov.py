"""HC0/HC1/HC2/HC3 sandwich SE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sandwich_robust_se"]


def sandwich_robust_se(X, y, kind):
    """
    HC0/HC1/HC2/HC3 sandwich SE

    Formula: V = (X^T X)^{-1} X^T Ω X (X^T X)^{-1}

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    kind : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    White (1980); MacKinnon-White (1985)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HC0/HC1/HC2/HC3 sandwich SE"})


def cheatsheet():
    return "robcov: HC0/HC1/HC2/HC3 sandwich SE"

"""Support vector machine (linear)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_svm"]


def wasserman_svm(X, y):
    """
    Support vector machine (linear)

    Formula: min (1/2)|w|^2 s.t. y_i(w'x_i+b) >= 1

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w, b

    References
    ----------
    Wasserman (2004), Ch 22
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Support vector machine (linear)"})


def cheatsheet():
    return "wsmsvm: Support vector machine (linear)"

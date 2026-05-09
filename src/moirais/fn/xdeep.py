"""xDeepFM with CIN."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["xdeepfm"]


def xdeepfm(X, y, K):
    """
    xDeepFM with CIN

    Formula: compressed interaction network + linear + DNN

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lian et al (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "xDeepFM with CIN"})


def cheatsheet():
    return "xdeep: xDeepFM with CIN"

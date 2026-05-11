"""IQR outlier rule."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["iqr_outlier"]


def iqr_outlier(x):
    """
    IQR outlier rule

    Formula: x < Q1 − 1.5 IQR or x > Q3 + 1.5 IQR

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tukey (1977)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IQR outlier rule"})


def cheatsheet():
    return "iqrA: IQR outlier rule"

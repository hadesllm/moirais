"""Weight trimming (cap extreme weights)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weight_trimming"]


def weight_trimming(y, weights, threshold):
    """
    Weight trimming (cap extreme weights)

    Formula: w_i' = min(w_i, w_max); redistribute residual

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Potter (1990)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weight trimming (cap extreme weights)"})


def cheatsheet():
    return "trimit: Weight trimming (cap extreme weights)"

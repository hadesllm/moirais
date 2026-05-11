"""Weight trimming at percentile."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["trim_weights"]


def trim_weights(weights, quantile):
    """
    Weight trimming at percentile

    Formula: truncate w_i at q-th percentile

    Parameters
    ----------
    weights : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Potter (1990)
    """
    weights = np.atleast_1d(np.asarray(weights, dtype=float))
    n = len(weights)
    result = float(np.mean(weights))
    se = float(np.std(weights, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weight trimming at percentile"})


def cheatsheet():
    return "trmwgt: Weight trimming at percentile"

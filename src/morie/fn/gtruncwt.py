"""Weight truncation for IPTW."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["truncate_weights"]


def truncate_weights(weights, quantile):
    """
    Weight truncation for IPTW

    Formula: w_trunc = min(w, q_99)

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
    Cole & Hernán (2008)
    """
    weights = np.atleast_1d(np.asarray(weights, dtype=float))
    n = len(weights)
    result = float(np.mean(weights))
    se = float(np.std(weights, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weight truncation for IPTW"})


def cheatsheet():
    return "gtruncwt: Weight truncation for IPTW"

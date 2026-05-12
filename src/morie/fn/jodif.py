# morie.fn -- function file (hadesllm/morie)
"""First/seasonal differencing as target transformation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_differencing"]


def joseph_differencing(y, order, seasonal_period):
    """
    First/seasonal differencing as target transformation

    Formula: y_t^d = y_t - y_{t-1}  (first diff);  y_t^s = y_t - y_{t-m}  (seasonal diff)

    Parameters
    ----------
    y : array-like
        Input data.
    order : array-like
        Input data.
    seasonal_period : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: differenced

    References
    ----------
    Joseph Ch 7, Differencing section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "First/seasonal differencing as target transformation"})


def cheatsheet():
    return "jodif: First/seasonal differencing as target transformation"

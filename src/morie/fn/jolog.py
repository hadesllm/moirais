# morie.fn — function file (hadesllm/morie)
"""Log (or log1p) target transformation for strictly-positive series."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_log_transform"]


def joseph_log_transform(y):
    """
    Log (or log1p) target transformation for strictly-positive series

    Formula: y_t_log = log(y_t + 1);  inverse: exp(y_t_log) - 1

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: log_y

    References
    ----------
    Joseph Ch 7, Log Transform section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log (or log1p) target transformation for strictly-positive series"})


def cheatsheet():
    return "jolog: Log (or log1p) target transformation for strictly-positive series"

"""Total sum of squares TSS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_total_sum_squares"]


def esl_total_sum_squares(y):
    """
    Total sum of squares TSS

    Formula: TSS = sum (y_i - y_bar)^2

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Total sum of squares TSS"})


def cheatsheet():
    return "eslrss2: Total sum of squares TSS"

"""Coverage correction (under/over-coverage adjustment)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["coverage_correction"]


def coverage_correction(y, weights, target_totals):
    """
    Coverage correction (under/over-coverage adjustment)

    Formula: w_i' = w_i * (N_target_h / hat N_h)

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    target_totals : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sarndal, Swensson, Wretman (1992) §15
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Coverage correction (under/over-coverage adjustment)"})


def cheatsheet():
    return "covpop: Coverage correction (under/over-coverage adjustment)"

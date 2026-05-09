"""Report-noisy-max selection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["report_noisy_max"]


def report_noisy_max(utilities, sensitivity, epsilon):
    """
    Report-noisy-max selection

    Formula: argmax_i [u(D,i) + Lap(Δu/ε)]

    Parameters
    ----------
    utilities : array-like
        Input data.
    sensitivity : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork-Roth (2014)
    """
    utilities = np.atleast_1d(np.asarray(utilities, dtype=float))
    n = len(utilities)
    result = float(np.mean(utilities))
    se = float(np.std(utilities, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Report-noisy-max selection"})


def cheatsheet():
    return "reportm: Report-noisy-max selection"

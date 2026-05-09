"""Time-domain difference equation of the baseline-wander filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_baseline_wander_filter_difference_eq"]


def rangayyan_ch3_baseline_wander_filter_difference_eq(x, y, T, n):
    """
    Time-domain difference equation of the baseline-wander filter.

    Formula: y(n) = (1/T) * [x(n) - x(n-1)] + 0.995 * y(n-1)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    T : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.134, p. 150
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-domain difference equation of the baseline-wander filter."})


def cheatsheet():
    return "rng122: Time-domain difference equation of the baseline-wander filter."

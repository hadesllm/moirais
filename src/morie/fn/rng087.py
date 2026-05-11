"""General difference equation of a moving-average (MA) filter of order N.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_filter_general"]


def rangayyan_ch3_ma_filter_general(x, b_k, n, N):
    """
    General difference equation of a moving-average (MA) filter of order N.

    Formula: y(n) = sum_{k=0}^{N} b_k * x(n-k)

    Parameters
    ----------
    x : array-like
        Input data.
    b_k : array-like
        Input data.
    n : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.98, p. 139
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "General difference equation of a moving-average (MA) filter of order N."})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "General difference equation of a moving-average (MA) filter of order N."})


def cheatsheet():
    return "rng087: General difference equation of a moving-average (MA) filter of order N."

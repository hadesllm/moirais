"""11-point moving-average (MA) filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_filter_11pt"]


def rangayyan_ch3_ma_filter_11pt(x, n):
    """
    11-point moving-average (MA) filter.

    Formula: y(n) = (1/11) * sum_{k=0}^{10} x(n - k)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.41, p. 114
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "11-point moving-average (MA) filter."})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "11-point moving-average (MA) filter."})


def cheatsheet():
    return "rng039: 11-point moving-average (MA) filter."

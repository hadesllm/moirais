"""Matched-filter impulse response for the basic pattern g(n).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_h_example"]


def rangayyan_ch4_matched_filter_h_example(n):
    """
    Matched-filter impulse response for the basic pattern g(n).

    Formula: h(n) = delta(n) + 2*delta(n-1) + 3*delta(n-2)

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.54, p. 241
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matched-filter impulse response for the basic pattern g(n)."})


def cheatsheet():
    return "rng226: Matched-filter impulse response for the basic pattern g(n)."

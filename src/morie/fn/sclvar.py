"""Selection coefficient estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["selection_coefficient"]


def selection_coefficient(freqs, generations, Ne):
    """
    Selection coefficient estimator

    Formula: s from allele frequency change over generations

    Parameters
    ----------
    freqs : array-like
        Input data.
    generations : array-like
        Input data.
    Ne : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Foll-Gaggiotti (2008)
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    n = len(freqs)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Selection coefficient estimator"})
    estimate = np.median(freqs)
    se = 1.2533 * np.std(freqs, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Selection coefficient estimator"})


def cheatsheet():
    return "sclvar: Selection coefficient estimator"

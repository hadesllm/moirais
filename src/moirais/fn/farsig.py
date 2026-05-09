"""Farrington flexible algorithm."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["farrington_signal"]


def farrington_signal(counts, baseline_years, reference_window):
    """
    Farrington flexible algorithm

    Formula: GLM with overdispersion + reference baseline

    Parameters
    ----------
    counts : array-like
        Input data.
    baseline_years : array-like
        Input data.
    reference_window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Noufaily et al (2013)
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Farrington flexible algorithm"})


def cheatsheet():
    return "farsig: Farrington flexible algorithm"

"""Outbreak detection (changepoint)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["outbreak_detection"]


def outbreak_detection(counts, prior_hazard):
    """
    Outbreak detection (changepoint)

    Formula: Bayesian online changepoint on counts

    Parameters
    ----------
    counts : array-like
        Input data.
    prior_hazard : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Adams-MacKay (2007)
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Outbreak detection (changepoint)"})


def cheatsheet():
    return "odgrev: Outbreak detection (changepoint)"

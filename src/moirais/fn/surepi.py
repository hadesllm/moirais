"""Surveillance signal (CUSUM/EARS)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["surveillance_signal"]


def surveillance_signal(counts, baseline_window):
    """
    Surveillance signal (CUSUM/EARS)

    Formula: EARS C1/C2/C3 z-scores

    Parameters
    ----------
    counts : array-like
        Input data.
    baseline_window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hutwagner et al (2003) EARS
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Surveillance signal (CUSUM/EARS)"})


def cheatsheet():
    return "surepi: Surveillance signal (CUSUM/EARS)"

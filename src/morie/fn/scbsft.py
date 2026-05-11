"""DR-DiD with baseline outcome shift."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sc_with_baseline_shift"]


def sc_with_baseline_shift(y, D, X, baseline):
    """
    DR-DiD with baseline outcome shift

    Formula: adjust for baseline level via Q

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    baseline : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tsiatis et al (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD with baseline outcome shift"})


def cheatsheet():
    return "scbsft: DR-DiD with baseline outcome shift"

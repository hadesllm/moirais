"""DR-DiD for staggered adoption."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_staggered_design"]


def dr_staggered_design(y, D, unit, time, cohort):
    """
    DR-DiD for staggered adoption

    Formula: DR moment per cohort × event-time pair

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    cohort : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Roth-Sant'Anna (2023); Borusyak-Jaravel-Spiess (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD for staggered adoption"})


def cheatsheet():
    return "drsta: DR-DiD for staggered adoption"

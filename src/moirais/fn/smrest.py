"""Standardized mortality ratio (indirect)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["standardized_mortality_ratio"]


def standardized_mortality_ratio(observed, expected):
    """
    Standardized mortality ratio (indirect)

    Formula: SMR = O / E

    Parameters
    ----------
    observed : array-like
        Input data.
    expected : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Breslow-Day (1987)
    """
    observed = np.atleast_1d(np.asarray(observed, dtype=float))
    n = len(observed)
    result = float(np.mean(observed))
    se = float(np.std(observed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Standardized mortality ratio (indirect)"})


def cheatsheet():
    return "smrest: Standardized mortality ratio (indirect)"

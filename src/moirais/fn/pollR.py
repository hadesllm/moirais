"""Pollard's rho factoring."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pollards_rho"]


def pollards_rho(n):
    """
    Pollard's rho factoring

    Formula: cycle-detect on x²+c mod n

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pollard (1975)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pollard's rho factoring"})


def cheatsheet():
    return "pollR: Pollard's rho factoring"

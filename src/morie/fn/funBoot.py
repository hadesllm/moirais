"""Functional bootstrap (curve-level)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_bootstrap"]


def functional_bootstrap(Y, n_boot):
    """
    Functional bootstrap (curve-level)

    Formula: resample curves with replacement

    Parameters
    ----------
    Y : array-like
        Input data.
    n_boot : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cuevas-Febrero-Fraiman (2006)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional bootstrap (curve-level)"})


def cheatsheet():
    return "funBoot: Functional bootstrap (curve-level)"

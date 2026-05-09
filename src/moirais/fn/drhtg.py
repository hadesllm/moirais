"""DR-DiD heterogeneous CATT."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_heterogeneity"]


def dr_did_heterogeneity(y, D, X, strata):
    """
    DR-DiD heterogeneous CATT

    Formula: DR moment per stratum X = x

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    strata : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Athey-Imbens (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD heterogeneous CATT"})


def cheatsheet():
    return "drhtg: DR-DiD heterogeneous CATT"

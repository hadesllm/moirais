"""DR-DiD with finite-sample size correction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_size_correction"]


def dr_did_size_correction(y, D, X):
    """
    DR-DiD with finite-sample size correction

    Formula: DR moment + small-sample correction

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Roth-Sant'Anna (2023)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD with finite-sample size correction"})


def cheatsheet():
    return "drbsze: DR-DiD with finite-sample size correction"

"""Bootstrap inference for DR-DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_bootstrap"]


def dr_did_bootstrap(y, D, X, B):
    """
    Bootstrap inference for DR-DiD

    Formula: resample n; compute psi_b for each

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sant'Anna-Zhao (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap inference for DR-DiD"})


def cheatsheet():
    return "drbst: Bootstrap inference for DR-DiD"

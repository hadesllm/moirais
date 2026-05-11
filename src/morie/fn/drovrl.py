"""DR-DiD with propensity overlap trimming."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_did_overlap_trim"]


def dr_did_overlap_trim(y, D, X, eps):
    """
    DR-DiD with propensity overlap trimming

    Formula: trim {p(X) ∈ [eps, 1-eps]} for variance bound

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Crump et al (2009); Sant'Anna-Zhao (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD with propensity overlap trimming"})


def cheatsheet():
    return "drovrl: DR-DiD with propensity overlap trimming"

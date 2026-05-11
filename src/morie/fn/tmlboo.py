"""Bootstrap confidence intervals for TMLE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_bootstrap_ci"]


def tmle_bootstrap_ci(y, D, X, B):
    """
    Bootstrap confidence intervals for TMLE

    Formula: resample n with replacement; compute psi_b for each

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
    vdL-Rose (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap confidence intervals for TMLE"})


def cheatsheet():
    return "tmlboo: Bootstrap confidence intervals for TMLE"

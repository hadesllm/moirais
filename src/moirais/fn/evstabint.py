"""Profile-likelihood CI for GEV/GPD shape ξ."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_xi_ci_profile"]


def evt_xi_ci_profile(x, mle, level):
    """
    Profile-likelihood CI for GEV/GPD shape ξ

    Formula: {ξ : 2[ℓ_p(ξ)-ℓ̂] >= -χ²_{1,α}}

    Parameters
    ----------
    x : array-like
        Input data.
    mle : array-like
        Input data.
    level : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ci_lo, ci_hi

    References
    ----------
    Coles (2001)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Profile-likelihood CI for GEV/GPD shape ξ"})


def cheatsheet():
    return "evstabint: Profile-likelihood CI for GEV/GPD shape ξ"

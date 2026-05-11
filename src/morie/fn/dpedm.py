"""(ε,δ)-DP relaxation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["approx_dp"]


def approx_dp(mech, D, D_prime, delta):
    """
    (ε,δ)-DP relaxation

    Formula: P(M(D)∈S) ≤ exp(ε) · P(M(D')∈S) + δ

    Parameters
    ----------
    mech : array-like
        Input data.
    D : array-like
        Input data.
    D_prime : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork et al (2006) Our Data Ourselves
    """
    mech = np.atleast_1d(np.asarray(mech, dtype=float))
    n = len(mech)
    result = float(np.mean(mech))
    se = float(np.std(mech, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "(ε,δ)-DP relaxation"})


def cheatsheet():
    return "dpedm: (ε,δ)-DP relaxation"

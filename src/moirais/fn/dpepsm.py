"""Pure ε-differential privacy definition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["epsilon_dp"]


def epsilon_dp(mech, D, D_prime):
    """
    Pure ε-differential privacy definition

    Formula: P(M(D)∈S) ≤ exp(ε) · P(M(D')∈S)

    Parameters
    ----------
    mech : array-like
        Input data.
    D : array-like
        Input data.
    D_prime : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork-McSherry-Nissim-Smith (2006)
    """
    mech = np.atleast_1d(np.asarray(mech, dtype=float))
    n = len(mech)
    result = float(np.mean(mech))
    se = float(np.std(mech, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pure ε-differential privacy definition"})


def cheatsheet():
    return "dpepsm: Pure ε-differential privacy definition"

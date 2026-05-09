"""f-DP / Gaussian DP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gaussian_dp"]


def gaussian_dp(mech, mu):
    """
    f-DP / Gaussian DP

    Formula: trade-off function T(P,Q)(α); μ-GDP

    Parameters
    ----------
    mech : array-like
        Input data.
    mu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dong-Roth-Su (2019)
    """
    mech = np.atleast_1d(np.asarray(mech, dtype=float))
    n = len(mech)
    result = float(np.mean(mech))
    se = float(np.std(mech, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "f-DP / Gaussian DP"})


def cheatsheet():
    return "gdpf: f-DP / Gaussian DP"

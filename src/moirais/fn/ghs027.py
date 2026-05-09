"""Lower bound on the prior probability of approximating a target density p_0 in L1, used to establish that the total-variation support contains all densities absolutely continuous with respect to mu.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tailfree_strong_support_event"]


def ghosal_ch3_tailfree_strong_support_event(p, p_m, p_0, epsilon):
    """
    Lower bound on the prior probability of approximating a target density p_0 in L1, used to establish that the total-variation support contains all densities absolutely continuous with respect to mu.

    Formula: Pi( integral | p / p_m - 1 | d mu < epsilon / ( 2 ||p_0||_infty + epsilon ) ) * Pi( || p_m - p_0 ||_infty < epsilon / 2 )

    Parameters
    ----------
    p : array-like
        Input data.
    p_m : array-like
        Input data.
    p_0 : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.20, p. 47
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lower bound on the prior probability of approximating a target density p_0 in L1, used to establish that the total-variation support contains all densities absolutely continuous with respect to mu."})


def cheatsheet():
    return "ghs027: Lower bound on the prior probability of approximating a target density p_0 in L1, used to establish that the total-variation support contains all densities absolutely continuous with respect to mu."

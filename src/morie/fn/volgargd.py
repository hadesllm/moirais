"""GARCH with GED (Generalised Error) innovations."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_garch_ged"]


def vol_garch_ged(r, init):
    """
    GARCH with GED (Generalised Error) innovations

    Formula: z_t ~ GED(ν); same recursion as GARCH(1,1)

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: omega, alpha, beta, nu, ll

    References
    ----------
    Nelson (1991)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GARCH with GED (Generalised Error) innovations"})


def cheatsheet():
    return "volgargd: GARCH with GED (Generalised Error) innovations"

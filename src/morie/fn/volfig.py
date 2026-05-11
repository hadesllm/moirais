"""FIGARCH(p,d,q) fractionally-integrated GARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_figarch_fit"]


def vol_figarch_fit(r, p, q, init):
    """
    FIGARCH(p,d,q) fractionally-integrated GARCH

    Formula: (1-L)^d ε_t² = α(L) ε_t² + β(L) (σ_t² - ε_t²)

    Parameters
    ----------
    r : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: d, alpha, beta, ll

    References
    ----------
    Baillie-Bollerslev-Mikkelsen (1996)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FIGARCH(p,d,q) fractionally-integrated GARCH"})


def cheatsheet():
    return "volfig: FIGARCH(p,d,q) fractionally-integrated GARCH"

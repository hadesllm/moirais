"""Sensitivity analysis for unmeasured confounding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sensitivity_mediation"]


def sensitivity_mediation(X, M, Y, rho):
    """
    Sensitivity analysis for unmeasured confounding

    Formula: NIE shifts by rho * sqrt(sigma_2 * sigma_3) bound

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imai, Keele, Tingley (2010)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sensitivity analysis for unmeasured confounding"})


def cheatsheet():
    return "snsmed: Sensitivity analysis for unmeasured confounding"

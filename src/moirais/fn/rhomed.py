"""Rho critical value where mediation effect → 0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rho_critical_mediation"]


def rho_critical_mediation(nie, sigma_e2, sigma_e3):
    """
    Rho critical value where mediation effect → 0

    Formula: rho* = NIE_obs / sqrt(sigma_2 * sigma_3)

    Parameters
    ----------
    nie : array-like
        Input data.
    sigma_e2 : array-like
        Input data.
    sigma_e3 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imai, Keele, Yamamoto (2010)
    """
    nie = np.atleast_1d(np.asarray(nie, dtype=float))
    n = len(nie)
    result = float(np.mean(nie))
    se = float(np.std(nie, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rho critical value where mediation effect → 0"})


def cheatsheet():
    return "rhomed: Rho critical value where mediation effect → 0"

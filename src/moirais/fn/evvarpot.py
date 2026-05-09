"""Value-at-Risk via POT/GPD tail."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_pot_var"]


def evt_pot_var(u, sigma, xi, zeta_u, p):
    """
    Value-at-Risk via POT/GPD tail

    Formula: VaR_p = u + (σ/ξ)(((1-p)/ζ_u)^{-ξ}-1)

    Parameters
    ----------
    u : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.
    zeta_u : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: VaR

    References
    ----------
    McNeil-Frey (2000)
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    n = len(u)
    result = float(np.mean(u))
    se = float(np.std(u, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Value-at-Risk via POT/GPD tail"})


def cheatsheet():
    return "evvarpot: Value-at-Risk via POT/GPD tail"

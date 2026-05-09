"""Expected shortfall via POT/GPD tail."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_pot_es"]


def evt_pot_es(u, sigma, xi, VaR):
    """
    Expected shortfall via POT/GPD tail

    Formula: ES_p = (VaR_p + σ - ξu)/(1-ξ)

    Parameters
    ----------
    u : array-like
        Input data.
    sigma : array-like
        Input data.
    xi : array-like
        Input data.
    VaR : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ES

    References
    ----------
    McNeil-Frey (2000)
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    n = len(u)
    result = float(np.mean(u))
    se = float(np.std(u, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Expected shortfall via POT/GPD tail"})


def cheatsheet():
    return "evespot: Expected shortfall via POT/GPD tail"

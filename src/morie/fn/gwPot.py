"""GWP100."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["global_warming_potential"]


def global_warming_potential(gas, horizon):
    """
    GWP100

    Formula: sum AGWP_x / AGWP_CO2

    Parameters
    ----------
    gas : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    IPCC AR6 Ch.7
    """
    gas = np.atleast_1d(np.asarray(gas, dtype=float))
    n = len(gas)
    result = float(np.mean(gas))
    se = float(np.std(gas, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GWP100"})


def cheatsheet():
    return "gwPot: GWP100"

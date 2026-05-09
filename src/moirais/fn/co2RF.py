"""CO₂ radiative forcing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["radiative_forcing_co2"]


def radiative_forcing_co2(C, C0):
    """
    CO₂ radiative forcing

    Formula: ΔF = 5.35 ln(C/C₀) W/m²

    Parameters
    ----------
    C : array-like
        Input data.
    C0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    IPCC AR6 (2021)
    """
    C = np.atleast_1d(np.asarray(C, dtype=float))
    n = len(C)
    result = float(np.mean(C))
    se = float(np.std(C, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CO₂ radiative forcing"})


def cheatsheet():
    return "co2RF: CO₂ radiative forcing"

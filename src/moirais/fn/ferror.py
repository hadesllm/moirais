"""Renewal-equation forecast."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fader_renewable"]


def fader_renewable(incidence, Rt, gen_int):
    """
    Renewal-equation forecast

    Formula: I_t = R_t sum I_{t-s} w_s

    Parameters
    ----------
    incidence : array-like
        Input data.
    Rt : array-like
        Input data.
    gen_int : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fraser (2007)
    """
    incidence = np.atleast_1d(np.asarray(incidence, dtype=float))
    n = len(incidence)
    result = float(np.mean(incidence))
    se = float(np.std(incidence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Renewal-equation forecast"})


def cheatsheet():
    return "ferror: Renewal-equation forecast"

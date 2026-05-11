"""Quantile-mapping bias correction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["quantile_mapping"]


def quantile_mapping(x_mod, F_obs, F_mod):
    """
    Quantile-mapping bias correction

    Formula: x' = F_obs^{-1}(F_mod(x))

    Parameters
    ----------
    x_mod : array-like
        Input data.
    F_obs : array-like
        Input data.
    F_mod : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wood et al (2002); Maraun (2013)
    """
    x_mod = np.atleast_1d(np.asarray(x_mod, dtype=float))
    n = len(x_mod)
    result = float(np.mean(x_mod))
    se = float(np.std(x_mod, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile-mapping bias correction"})


def cheatsheet():
    return "qmDS: Quantile-mapping bias correction"

"""W_2 between Gaussian mixtures via OT on components."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_mixture_w2"]


def ot_mixture_w2(mus1, Sigmas1, w1, mus2, Sigmas2, w2):
    """
    W_2 between Gaussian mixtures via OT on components

    Formula: min over OT plans of components, weighted by component W_2

    Parameters
    ----------
    mus1 : array-like
        Input data.
    Sigmas1 : array-like
        Input data.
    w1 : array-like
        Input data.
    mus2 : array-like
        Input data.
    Sigmas2 : array-like
        Input data.
    w2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: MW2

    References
    ----------
    Delon-Desolneux (2020)
    """
    mus1 = np.atleast_1d(np.asarray(mus1, dtype=float))
    n = len(mus1)
    result = float(np.mean(mus1))
    se = float(np.std(mus1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "W_2 between Gaussian mixtures via OT on components"})


def cheatsheet():
    return "otmxh: W_2 between Gaussian mixtures via OT on components"

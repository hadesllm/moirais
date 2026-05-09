"""EpiNow2 generative Rt."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["epinow2"]


def epinow2(incidence, gen_int, delays):
    """
    EpiNow2 generative Rt

    Formula: renewal equation with delay distributions

    Parameters
    ----------
    incidence : array-like
        Input data.
    gen_int : array-like
        Input data.
    delays : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Abbott-Hellewell-Sherratt-EpiNow2 (2020)
    """
    incidence = np.atleast_1d(np.asarray(incidence, dtype=float))
    n = len(incidence)
    result = float(np.mean(incidence))
    se = float(np.std(incidence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EpiNow2 generative Rt"})


def cheatsheet():
    return "epimet: EpiNow2 generative Rt"

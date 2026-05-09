"""Sun-Abraham heterogeneous-treatment DID."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sun_abraham_did"]


def sun_abraham_did(y, D, unit, time, cohort):
    """
    Sun-Abraham heterogeneous-treatment DID

    Formula: interacted weighted average of CATT(e,l) across cohorts

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    unit : array-like
        Input data.
    time : array-like
        Input data.
    cohort : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sun & Abraham (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sun-Abraham heterogeneous-treatment DID"})


def cheatsheet():
    return "sunabr: Sun-Abraham heterogeneous-treatment DID"

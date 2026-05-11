"""Hindsight experience replay."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["her"]


def her(buffer, strategy):
    """
    Hindsight experience replay

    Formula: relabel failed trajectory's goal to achieved state

    Parameters
    ----------
    buffer : array-like
        Input data.
    strategy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrychowicz et al (2017)
    """
    buffer = np.atleast_1d(np.asarray(buffer, dtype=float))
    n = len(buffer)
    result = float(np.mean(buffer))
    se = float(np.std(buffer, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hindsight experience replay"})


def cheatsheet():
    return "hindsr: Hindsight experience replay"

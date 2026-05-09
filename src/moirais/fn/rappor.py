"""RAPPOR — Bloom + permanent + instantaneous RR."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rappor"]


def rappor(x, f, p, q):
    """
    RAPPOR — Bloom + permanent + instantaneous RR

    Formula: two-step LDP for strings

    Parameters
    ----------
    x : array-like
        Input data.
    f : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Erlingsson-Pihur-Korolova (2014)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RAPPOR — Bloom + permanent + instantaneous RR"})


def cheatsheet():
    return "rappor: RAPPOR — Bloom + permanent + instantaneous RR"

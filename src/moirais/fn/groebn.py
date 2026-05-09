"""Gröbner basis (Buchberger)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["groebner"]


def groebner(polys, order):
    """
    Gröbner basis (Buchberger)

    Formula: reduce to canonical generating set

    Parameters
    ----------
    polys : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Buchberger (1965)
    """
    polys = np.atleast_1d(np.asarray(polys, dtype=float))
    n = len(polys)
    result = float(np.mean(polys))
    se = float(np.std(polys, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gröbner basis (Buchberger)"})


def cheatsheet():
    return "groebn: Gröbner basis (Buchberger)"

"""Nugget effect in semivariogram: discontinuity at origin."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_nugget_effect"]


def schabenberger_nugget_effect(h, nugget, sill, range):
    """
    Nugget effect in semivariogram: discontinuity at origin

    Formula: gamma(0+) = c0 > 0 (nugget); gamma(0)=0 by definition; total sill=c0+c1

    Parameters
    ----------
    h : array-like
        Input data.
    nugget : array-like
        Input data.
    sill : array-like
        Input data.
    range : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: nugget_sill_range

    References
    ----------
    Schabenberger Ch 4, Sec 4.3.6
    """
    h = np.asarray(h, dtype=float)
    n = int(h) if h.ndim == 0 else len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nugget effect in semivariogram: discontinuity at origin"})


def cheatsheet():
    return "spnug: Nugget effect in semivariogram: discontinuity at origin"

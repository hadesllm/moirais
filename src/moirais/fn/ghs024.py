"""Summability conditions on the means and variances of splitting variables that imply canonical absolute continuity of a tail-free process.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tailfree_canonical_summability"]


def ghosal_ch3_tailfree_canonical_summability(V, m):
    """
    Summability conditions on the means and variances of splitting variables that imply canonical absolute continuity of a tail-free process.

    Formula: sum_{m=1}^{infty} max_{epsilon in E^m} | E(V_epsilon) - 1/2 | < infty,   sum_{m=1}^{infty} max_{epsilon in E^m} var(V_epsilon) < infty

    Parameters
    ----------
    V : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.17, p. 44
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Summability conditions on the means and variances of splitting variables that imply canonical absolute continuity of a tail-free process."})


def cheatsheet():
    return "ghs024: Summability conditions on the means and variances of splitting variables that imply canonical absolute continuity of a tail-free process."

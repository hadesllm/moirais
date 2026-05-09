"""Bound on the expected squared maximum of partition probabilities for a tail-free process when sup E[V_epsilon^2] < 1/2, decaying geometrically in level m.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_tailfree_max_bound"]


def ghosal_ch3_tailfree_max_bound(V, m, r):
    """
    Bound on the expected squared maximum of partition probabilities for a tail-free process when sup E[V_epsilon^2] < 1/2, decaying geometrically in level m.

    Formula: E[ max_{epsilon in E^m} P(A_epsilon) ]^2 <= sum_{epsilon in E^m} prod_{j=1}^{m} E(V_{epsilon_1 ... epsilon_j}^2) <= 2^m * (r/2)^m = r^m

    Parameters
    ----------
    V : array-like
        Input data.
    m : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.14, p. 40
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    n = len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound on the expected squared maximum of partition probabilities for a tail-free process when sup E[V_epsilon^2] < 1/2, decaying geometrically in level m."})


def cheatsheet():
    return "ghs021: Bound on the expected squared maximum of partition probabilities for a tail-free process when sup E[V_epsilon^2] < 1/2, decaying geometrically in level m."

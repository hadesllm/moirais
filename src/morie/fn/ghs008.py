"""Construction of a prior on the infinite simplex by normalizing positive random variables Y_j to obtain probability weights p_k.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_normalized_weights_prior"]


def ghosal_ch3_normalized_weights_prior(Y_j, k):
    """
    Construction of a prior on the infinite simplex by normalizing positive random variables Y_j to obtain probability weights p_k.

    Formula: p_k = Y_k / sum_{j=1}^{infty} Y_j,   k in N

    Parameters
    ----------
    Y_j : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.1, p. 29
    """
    Y_j = np.atleast_1d(np.asarray(Y_j, dtype=float))
    n = len(Y_j)
    result = float(np.mean(Y_j))
    se = float(np.std(Y_j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Construction of a prior on the infinite simplex by normalizing positive random variables Y_j to obtain probability weights p_k."})


def cheatsheet():
    return "ghs008: Construction of a prior on the infinite simplex by normalizing positive random variables Y_j to obtain probability weights p_k."

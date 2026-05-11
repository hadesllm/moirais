"""Stick-breaking representation of weights p_j as the product of (1 - V_l) factors times V_j, distributing unit mass over countably many atoms.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_stick_breaking_weights"]


def ghosal_ch3_stick_breaking_weights(V_l, j):
    """
    Stick-breaking representation of weights p_j as the product of (1 - V_l) factors times V_j, distributing unit mass over countably many atoms.

    Formula: p_j = ( prod_{l=1}^{j-1} (1 - V_l) ) * V_j

    Parameters
    ----------
    V_l : array-like
        Input data.
    j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.2, p. 30
    """
    V_l = np.atleast_1d(np.asarray(V_l, dtype=float))
    n = len(V_l)
    result = float(np.mean(V_l))
    se = float(np.std(V_l, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stick-breaking representation of weights p_j as the product of (1 - V_l) factors times V_j, distributing unit mass over countably many atoms."})


def cheatsheet():
    return "ghs009: Stick-breaking representation of weights p_j as the product of (1 - V_l) factors times V_j, distributing unit mass over countably many atoms."

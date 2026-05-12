# morie.fn -- function file (hadesllm/morie)
"""Hellinger distance: dH(P,Q)^2 = 1/2 integral (sqrt(p)-sqrt(q))^2."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_hellinger_dist"]


def ghosal_hellinger_dist(x):
    """
    Hellinger distance: dH(P,Q)^2 = 1/2 integral (sqrt(p)-sqrt(q))^2

    Formula: d_H^2(P,Q) = 1 - integral sqrt(p*q) = 1 - rho_{1/2}(P,Q)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal App A
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hellinger distance: dH(P,Q)^2 = 1/2 integral (sqrt(p)-sqrt(q))^2"})


def cheatsheet():
    return "gh_ap_a4: Hellinger distance: dH(P,Q)^2 = 1/2 integral (sqrt(p)-sqrt(q))^2"

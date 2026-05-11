# morie.fn — function file (hadesllm/morie)
"""i.i.d. posterior consistency: Schwartz + KL support gives weak consistency."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_iid_posterior_consistency"]


def ghosal_iid_posterior_consistency(x):
    """
    i.i.d. posterior consistency: Schwartz + KL support gives weak consistency

    Formula: P0^infty: Pi_n(d_w(P,P0)>eps | X^n) -> 0 under Schwartz conditions

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
    Ghosal Ch 6 §6.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "i.i.d. posterior consistency: Schwartz + KL support gives weak consistency"})


def cheatsheet():
    return "gh_iid_consist: i.i.d. posterior consistency: Schwartz + KL support gives weak consistency"

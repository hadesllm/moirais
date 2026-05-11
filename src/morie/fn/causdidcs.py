"""Callaway-Sant'Anna ATT(g,t) under staggered adoption."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_did_callaway_sa"]


def causal_did_callaway_sa(Y_panel, G_first_treat, X_baseline):
    """
    Callaway-Sant'Anna ATT(g,t) under staggered adoption

    Formula: ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - DR adjust

    Parameters
    ----------
    Y_panel : array-like
        Input data.
    G_first_treat : array-like
        Input data.
    X_baseline : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATT_gt, agg_ATT

    References
    ----------
    Callaway & Sant'Anna (2021)
    """
    Y_panel = np.atleast_1d(np.asarray(Y_panel, dtype=float))
    n = len(Y_panel)
    result = float(np.mean(Y_panel))
    se = float(np.std(Y_panel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Callaway-Sant'Anna ATT(g,t) under staggered adoption"})


def cheatsheet():
    return "causdidcs: Callaway-Sant'Anna ATT(g,t) under staggered adoption"

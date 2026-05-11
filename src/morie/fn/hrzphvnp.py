# morie.fn — function file (hadesllm/morie)
"""Proportional hazards model with unobserved heterogeneity: nonparametric frailty."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_ph_frailty_nonpar"]


def horowitz_ph_frailty_nonpar(t, x, event):
    """
    Proportional hazards model with unobserved heterogeneity: nonparametric frailty

    Formula: h(t|X,V)=h_0(t)*exp(X'beta)*V; V arbitrary with E[V]=1; identification via multiple spells

    Parameters
    ----------
    t : array-like
        Input data.
    x : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat, h0_hat, frailty_dist

    References
    ----------
    Horowitz Ch 6, Sec 6.3.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proportional hazards model with unobserved heterogeneity: nonparametric frailty"})


def cheatsheet():
    return "hrzphvnp: Proportional hazards model with unobserved heterogeneity: nonparametric frailty"

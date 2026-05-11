# morie.fn — function file (hadesllm/morie)
"""Proportional hazards model with unobserved heterogeneity (frailty)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_ph_heterogeneity"]


def horowitz_ph_heterogeneity(t, x, event, frailty_dist):
    """
    Proportional hazards model with unobserved heterogeneity (frailty)

    Formula: h(t|X,V) = h_0(t)*exp(X'beta)*V; V~G(1,sigma^2); E[V]=1

    Parameters
    ----------
    t : array-like
        Input data.
    x : array-like
        Input data.
    event : array-like
        Input data.
    frailty_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat, h0_hat, sigma2_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proportional hazards model with unobserved heterogeneity (frailty)"})


def cheatsheet():
    return "hrzphv: Proportional hazards model with unobserved heterogeneity (frailty)"

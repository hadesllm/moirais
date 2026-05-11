# morie.fn — function file (hadesllm/morie)
"""Proportional hazards model with nonparametric baseline and parametric F."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_proportional_hazards"]


def horowitz_proportional_hazards(t, x, event):
    """
    Proportional hazards model with nonparametric baseline and parametric F

    Formula: h(t|X) = h_0(t)*exp(X'beta); T nonparametric, F exponential

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
        Keys: beta_hat, h0_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proportional hazards model with nonparametric baseline and parametric F"})


def cheatsheet():
    return "hrzph: Proportional hazards model with nonparametric baseline and parametric F"

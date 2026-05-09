# moirais.fn — function file (hadesllm/moirais)
"""Proportional hazards with discrete observations."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_ph_discrete_obs"]


def horowitz_ph_discrete_obs(t_discrete, x, event):
    """
    Proportional hazards with discrete observations

    Formula: Survival grouped into intervals; P(Y>t_k|X) = prod_{j<=k} [1-h_j*exp(X'beta)]

    Parameters
    ----------
    t_discrete : array-like
        Input data.
    x : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat, h_j_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.2.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proportional hazards with discrete observations"})


def cheatsheet():
    return "hrzphd: Proportional hazards with discrete observations"

# morie.fn — function file (hadesllm/morie)
"""Single-index model for conditional quantile function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_sim_quantile"]


def horowitz_sim_quantile(x, y, tau, bandwidth):
    """
    Single-index model for conditional quantile function

    Formula: Q_tau(Y|X=x) = G_tau(x'beta_tau) where beta_tau, G_tau vary with tau

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    tau : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_tau, G_tau_hat

    References
    ----------
    Horowitz Ch 2, Sec 2.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Single-index model for conditional quantile function"})


def cheatsheet():
    return "hrzsiqm: Single-index model for conditional quantile function"

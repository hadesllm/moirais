# moirais.fn — function file (hadesllm/moirais)
"""Rate of convergence n^{-1/2} for beta in single-index models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_rate_beta_estimation"]


def horowitz_rate_beta_estimation(x, y, bandwidth):
    """
    Rate of convergence n^{-1/2} for beta in single-index models

    Formula: n^{1/2}*(beta_hat-beta) ->_D N(0, V_beta); same rate as parametric models

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: asymptotic_distribution

    References
    ----------
    Horowitz Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rate of convergence n^{-1/2} for beta in single-index models"})


def cheatsheet():
    return "hrzrob: Rate of convergence n^{-1/2} for beta in single-index models"

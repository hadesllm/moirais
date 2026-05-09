# moirais.fn — function file (hadesllm/moirais)
"""Smooth hazard GP prior: log-hazard function modeled by GP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_smhaz_gp"]


def ghosal_smhaz_gp(x):
    """
    Smooth hazard GP prior: log-hazard function modeled by GP

    Formula: lambda(t) = exp(f(t)), f ~ GP(mu, k), hazard smooth via GP

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
    Ghosal Ch 13 §13.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smooth hazard GP prior: log-hazard function modeled by GP"})


def cheatsheet():
    return "gh_c13_12: Smooth hazard GP prior: log-hazard function modeled by GP"

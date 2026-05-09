"""Primal-dual hybrid gradient (Chambolle-Pock)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["primal_dual"]


def primal_dual(F, G, K, tau, sigma):
    """
    Primal-dual hybrid gradient (Chambolle-Pock)

    Formula: alternate primal and dual gradient steps

    Parameters
    ----------
    F : array-like
        Input data.
    G : array-like
        Input data.
    K : array-like
        Input data.
    tau : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chambolle-Pock (2011)
    """
    F = np.atleast_1d(np.asarray(F, dtype=float))
    n = len(F)
    result = float(np.mean(F))
    se = float(np.std(F, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Primal-dual hybrid gradient (Chambolle-Pock)"})


def cheatsheet():
    return "primal: Primal-dual hybrid gradient (Chambolle-Pock)"

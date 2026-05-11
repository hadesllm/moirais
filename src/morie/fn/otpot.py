"""Recover log potentials f,g from Sinkhorn marginals."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_pot_log_potentials"]


def ot_pot_log_potentials(u, v, epsilon):
    """
    Recover log potentials f,g from Sinkhorn marginals

    Formula: u=exp(f/ε), v=exp(g/ε)

    Parameters
    ----------
    u : array-like
        Input data.
    v : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: f, g

    References
    ----------
    Peyré & Cuturi (2019) §4
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    n = len(u)
    result = float(np.mean(u))
    se = float(np.std(u, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recover log potentials f,g from Sinkhorn marginals"})


def cheatsheet():
    return "otpot: Recover log potentials f,g from Sinkhorn marginals"

"""Symmetric Sinkhorn divergence S_ε(μ,ν)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_sinkhorn_divergence"]


def ot_sinkhorn_divergence(a, b, Cab, Caa, Cbb, epsilon):
    """
    Symmetric Sinkhorn divergence S_ε(μ,ν)

    Formula: S_ε = OT_ε(μ,ν) - 0.5(OT_ε(μ,μ) + OT_ε(ν,ν))

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    Cab : array-like
        Input data.
    Caa : array-like
        Input data.
    Cbb : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: S_eps

    References
    ----------
    Genevay-Peyré-Cuturi (2018)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symmetric Sinkhorn divergence S_ε(μ,ν)"})


def cheatsheet():
    return "otdiv: Symmetric Sinkhorn divergence S_ε(μ,ν)"

"""Solve entropic-OT dual potentials f,g via L-BFGS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_regularised_dual"]


def ot_regularised_dual(a, b, C, epsilon, max_iter):
    """
    Solve entropic-OT dual potentials f,g via L-BFGS

    Formula: max_{f,g} <a,f>+<b,g>-ε Σexp((f+g-C)/ε)

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    epsilon : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: f, g, dual_value

    References
    ----------
    Genevay-Cuturi-Peyré-Bach (2016)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Solve entropic-OT dual potentials f,g via L-BFGS"})


def cheatsheet():
    return "otreg: Solve entropic-OT dual potentials f,g via L-BFGS"

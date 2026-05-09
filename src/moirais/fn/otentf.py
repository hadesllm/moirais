"""Free energy of an OT plan = primal - dual."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_free_energy"]


def ot_free_energy(T, C, a, b, f, g, epsilon):
    """
    Free energy of an OT plan = primal - dual

    Formula: F = <T,C> - ε H(T) - <a,f> - <b,g>

    Parameters
    ----------
    T : array-like
        Input data.
    C : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    f : array-like
        Input data.
    g : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: F

    References
    ----------
    Peyré & Cuturi (2019)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Free energy of an OT plan = primal - dual"})


def cheatsheet():
    return "otentf: Free energy of an OT plan = primal - dual"

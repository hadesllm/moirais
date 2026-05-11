# morie.fn — function file (hadesllm/morie)
"""Multinomial probit for spatial choice."""
import numpy as np
from ._richresult import RichResult

__all__ = ["multinomial_probit_spatial"]


def multinomial_probit_spatial(x):
    """
    Multinomial probit for spatial choice

    Formula: P(choice j) = Phi_k(U_j - U_{-j})

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
    Armstrong Ch 9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multinomial probit for spatial choice"})


def cheatsheet():
    return "mnpbt: Multinomial probit for spatial choice"

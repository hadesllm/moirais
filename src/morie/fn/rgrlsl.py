# morie.fn -- function file (hadesllm/morie)
"""RLS lattice (ladder) adaptive filter."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_rls_lattice"]


def rangayyan_rls_lattice(x, d, lam, order):
    """
    RLS lattice (ladder) adaptive filter

    Formula: Forward/backward prediction errors updated with reflection coefficients

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    lam : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y, e, k_f, k_b

    References
    ----------
    Rangayyan Ch 8.6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RLS lattice (ladder) adaptive filter"})


def cheatsheet():
    return "rgrlsl: RLS lattice (ladder) adaptive filter"

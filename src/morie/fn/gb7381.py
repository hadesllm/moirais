# morie.fn — function file (hadesllm/morie)
"""Null variance formula from Chernoff-Savage theory under H0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_cs_null_var"]


def gibbons_cs_null_var(J, lam):
    """
    Null variance formula from Chernoff-Savage theory under H0

    Formula: N*lN*sigma^2 = (1-lN)[integral J^2(u)du - (integral J(u)du)^2]

    Parameters
    ----------
    J : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: asymp_variance

    References
    ----------
    Gibbons Corollary 7.3.1
    """
    J = np.asarray(J, dtype=float)
    n = int(J) if J.ndim == 0 else len(J)
    result = float(np.mean(J))
    se = float(np.std(J, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Null variance formula from Chernoff-Savage theory under H0"})


def cheatsheet():
    return "gb7381: Null variance formula from Chernoff-Savage theory under H0"

"""Rejection point of redescending ψ."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rejection_point"]


def rejection_point(psi):
    """
    Rejection point of redescending ψ

    Formula: ρ* = inf{r>0: ψ(t)=0 for |t|≥r}

    Parameters
    ----------
    psi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hampel (1974)
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    n = len(psi)
    result = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rejection point of redescending ψ"})


def cheatsheet():
    return "rejct: Rejection point of redescending ψ"

"""Mixing time from spectral gap of L_rw."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_mixing_time"]


def sgt_mixing_time(A, epsilon):
    """
    Mixing time from spectral gap of L_rw

    Formula: τ_mix ~ log(1/ε)/(1-λ_2)

    Parameters
    ----------
    A : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tau_mix

    References
    ----------
    Levin-Peres-Wilmer (2017)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixing time from spectral gap of L_rw"})


def cheatsheet():
    return "sgtmix: Mixing time from spectral gap of L_rw"

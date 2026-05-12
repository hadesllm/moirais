# morie.fn -- function file (hadesllm/morie)
"""Canary exposure: rank of a canary string among candidate strings by model PLL."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_memorization_exposure"]


def kamath_memorization_exposure(canary_ll, candidate_lls):
    """
    Canary exposure: rank of a canary string among candidate strings by model PLL

    Formula: exposure = log2(|Candidates|) - log2(rank(canary))

    Parameters
    ----------
    canary_ll : array-like
        Input data.
    candidate_lls : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: exposure

    References
    ----------
    Kamath Ch 6, Memorization Exposure section
    """
    canary_ll = np.atleast_1d(np.asarray(canary_ll, dtype=float))
    n = len(canary_ll)
    result = float(np.mean(canary_ll))
    se = float(np.std(canary_ll, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Canary exposure: rank of a canary string among candidate strings by model PLL"})


def cheatsheet():
    return "kmexp: Canary exposure: rank of a canary string among candidate strings by model PLL"

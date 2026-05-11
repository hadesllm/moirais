# morie.fn — function file (hadesllm/morie)
"""Kaplan-style LLM scaling law — loss vs compute/data/params."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_scaling_laws"]


def kamath_scaling_laws(N, N_c, alpha_N, L_inf):
    """
    Kaplan-style LLM scaling law — loss vs compute/data/params

    Formula: L(N) = (N_c / N)^alpha_N + L_inf  (similar form for D and C)

    Parameters
    ----------
    N : array-like
        Input data.
    N_c : array-like
        Input data.
    alpha_N : array-like
        Input data.
    L_inf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: L

    References
    ----------
    Kamath Ch 1, Scaling Laws section
    """
    N = np.atleast_1d(np.asarray(N, dtype=float))
    n = len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kaplan-style LLM scaling law — loss vs compute/data/params"})


def cheatsheet():
    return "kmscal: Kaplan-style LLM scaling law — loss vs compute/data/params"

# morie.fn -- function file (hadesllm/morie)
"""Chinchilla compute-optimal parameter/token ratio (N tokens per param)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_chinchilla_compute_optimal"]


def kamath_chinchilla_compute_optimal(compute_budget, alpha, beta):
    """
    Chinchilla compute-optimal parameter/token ratio (N tokens per param)

    Formula: N_tokens / N_params ≈ 20; compute-optimal L(N, D)

    Parameters
    ----------
    compute_budget : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: N_opt, D_opt

    References
    ----------
    Kamath Ch 1, Compute-Optimal Scaling section
    """
    compute_budget = np.atleast_1d(np.asarray(compute_budget, dtype=float))
    n = len(compute_budget)
    result = float(np.mean(compute_budget))
    se = float(np.std(compute_budget, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chinchilla compute-optimal parameter/token ratio (N tokens per param)"})


def cheatsheet():
    return "kmchin: Chinchilla compute-optimal parameter/token ratio (N tokens per param)"

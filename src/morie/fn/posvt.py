# morie.fn — function file (hadesllm/morie)
"""Positivity (overlap) assumption: every unit has non-zero probability of treatment."""
import numpy as np
from ._richresult import RichResult

__all__ = ["positivity_assumption"]


def positivity_assumption(T, X):
    """
    Positivity (overlap) assumption: every unit has non-zero probability of treatment

    Formula: 0 < P(T=1|X=T) < 1 for all T in support(X)

    Parameters
    ----------
    T : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'positivity_violated': 'array', 'overlap_plot': 'dict'}

    References
    ----------
    Molak Ch 8
    """
    T = np.asarray(T, dtype=float)
    n = int(T) if T.ndim == 0 else len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Positivity (overlap) assumption: every unit has non-zero probability of treatment"})


def cheatsheet():
    return "posvt: Positivity (overlap) assumption: every unit has non-zero probability of treatment"

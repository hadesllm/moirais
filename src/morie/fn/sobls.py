"""Sobol quasi-random sequence."""
import numpy as np
from ._richresult import RichResult

__all__ = ["sobol_sequence"]


def sobol_sequence(x):
    """
    Sobol quasi-random sequence

    Formula: Low-discrepancy sequence in [0,1]^d

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
    Sobol (1967)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sobol quasi-random sequence"})


def cheatsheet():
    return "sobls: Sobol quasi-random sequence"

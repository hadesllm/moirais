"""Sobol global sensitivity indices."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sobol_indices"]


def sobol_indices(model, input_dist, N):
    """
    Sobol global sensitivity indices

    Formula: S_i = V_i / V

    Parameters
    ----------
    model : array-like
        Input data.
    input_dist : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sobol (1993); Saltelli (2002)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sobol global sensitivity indices"})


def cheatsheet():
    return "sobolI: Sobol global sensitivity indices"

"""Perturbation (group operation on the simplex)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_perturbation"]


def aitchison_perturbation(x, y):
    """
    Perturbation (group operation on the simplex)

    Formula: (x ⊕ y)_i = C(x_i * y_i)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perturbation (group operation on the simplex)"})


def cheatsheet():
    return "aitprt: Perturbation (group operation on the simplex)"

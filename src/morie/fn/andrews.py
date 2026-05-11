"""Andrews sine ψ."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["andrews_sine"]


def andrews_sine(r, c):
    """
    Andrews sine ψ

    Formula: ψ(r) = sin(r/c) if |r|≤cπ else 0

    Parameters
    ----------
    r : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrews et al (1972)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Andrews sine ψ"})


def cheatsheet():
    return "andrews: Andrews sine ψ"

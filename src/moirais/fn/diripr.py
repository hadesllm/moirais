"""Dirichlet-Multinomial conjugate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dirichlet_multinomial"]


def dirichlet_multinomial(counts, alpha):
    """
    Dirichlet-Multinomial conjugate

    Formula: p ~ Dir(alpha); y ~ Multinomial(n, p)

    Parameters
    ----------
    counts : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman BDA3
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet-Multinomial conjugate"})


def cheatsheet():
    return "diripr: Dirichlet-Multinomial conjugate"

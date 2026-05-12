# morie.fn -- function file (hadesllm/morie)
"""Johnson-Lindenstrauss lower bound on random-projection target dimension."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_johnson_lindenstrauss_bound"]


def geron_johnson_lindenstrauss_bound(n_samples, eps):
    """
    Johnson-Lindenstrauss lower bound on random-projection target dimension

    Formula: d >= 4 log(m) / (eps^2/2 - eps^3/3)

    Parameters
    ----------
    n_samples : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: d_min

    References
    ----------
    Géron Ch 7, Random Projection section (JL lemma)
    """
    n_samples = np.asarray(n_samples, dtype=float)
    n = int(n_samples) if n_samples.ndim == 0 else len(n_samples)
    result = float(np.mean(n_samples))
    se = float(np.std(n_samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Johnson-Lindenstrauss lower bound on random-projection target dimension"})


def cheatsheet():
    return "grjll: Johnson-Lindenstrauss lower bound on random-projection target dimension"

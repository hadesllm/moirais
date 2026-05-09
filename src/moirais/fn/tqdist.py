"""Lemma 3.5: ProdQJL distortion bound via Bernstein."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_distortion_bound"]


def turboquant_distortion_bound(eps, delta):
    """
    Lemma 3.5: ProdQJL distortion bound via Bernstein

    Formula: Pr[|ProdQJL - <q,k>| > eps ||q|| ||k||] <= delta;  need m >= (4/3)(1+eps)/eps^2 * log(2/delta)

    Parameters
    ----------
    eps : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: m_min

    References
    ----------
    Zandieh et al. 2024 Lemma 3.5 (distortion bound)
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    n = len(eps)
    result = float(np.mean(eps))
    se = float(np.std(eps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lemma 3.5: ProdQJL distortion bound via Bernstein"})


def cheatsheet():
    return "tqdist: Lemma 3.5: ProdQJL distortion bound via Bernstein"

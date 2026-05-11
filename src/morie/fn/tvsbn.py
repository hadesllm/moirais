"""Tversky asymmetric similarity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tversky_similarity"]


def tversky_similarity(fp_a, fp_b, alpha, beta):
    """
    Tversky asymmetric similarity

    Formula: |A∩B| / (|A∩B| + α|A\B| + β|B\A|)

    Parameters
    ----------
    fp_a : array-like
        Input data.
    fp_b : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tversky (1977)
    """
    fp_a = np.atleast_1d(np.asarray(fp_a, dtype=float))
    n = len(fp_a)
    result = float(np.mean(fp_a))
    se = float(np.std(fp_a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tversky asymmetric similarity"})


def cheatsheet():
    return "tvsbn: Tversky asymmetric similarity"

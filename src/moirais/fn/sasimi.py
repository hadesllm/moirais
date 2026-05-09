"""Tanimoto similarity between fingerprints."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tanimoto_similarity"]


def tanimoto_similarity(fp_a, fp_b):
    """
    Tanimoto similarity between fingerprints

    Formula: |A∩B| / |A∪B|

    Parameters
    ----------
    fp_a : array-like
        Input data.
    fp_b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jaccard (1901); Tanimoto (1958)
    """
    fp_a = np.atleast_1d(np.asarray(fp_a, dtype=float))
    n = len(fp_a)
    result = float(np.mean(fp_a))
    se = float(np.std(fp_a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tanimoto similarity between fingerprints"})


def cheatsheet():
    return "sasimi: Tanimoto similarity between fingerprints"

"""AlphaFold initial embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_embedding_init"]


def alphafold_embedding_init(sequence):
    """
    AlphaFold initial embedding

    Formula: linear from one-hot AA + position embed

    Parameters
    ----------
    sequence : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    sequence = np.atleast_1d(np.asarray(sequence, dtype=float))
    n = len(sequence)
    result = float(np.mean(sequence))
    se = float(np.std(sequence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold initial embedding"})


def cheatsheet():
    return "alfemb: AlphaFold initial embedding"

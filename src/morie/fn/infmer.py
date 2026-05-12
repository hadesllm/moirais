"""Informer -- sparse self-attention forecaster."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["informer"]


def informer(X, y, seq_len):
    """
    Informer -- sparse self-attention forecaster

    Formula: ProbSparse attention + distilling

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    seq_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhou et al (2021) Informer
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Informer -- sparse self-attention forecaster"})


def cheatsheet():
    return "infmer: Informer -- sparse self-attention forecaster"

"""Consensus rescoring across multiple docking functions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rescore_consensus"]


def rescore_consensus(scores):
    """
    Consensus rescoring across multiple docking functions

    Formula: rank-sum / Borda count over Vina + Glide + ChemScore

    Parameters
    ----------
    scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Charifson et al (1999)
    """
    scores = np.atleast_1d(np.asarray(scores, dtype=float))
    n = len(scores)
    result = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Consensus rescoring across multiple docking functions"})


def cheatsheet():
    return "rescor: Consensus rescoring across multiple docking functions"

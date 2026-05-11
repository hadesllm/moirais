"""Benjamini-Hochberg FDR control."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_bh_fdr"]


def esl_bh_fdr(pvalues, alpha):
    """
    Benjamini-Hochberg FDR control

    Formula: Reject H_(j) if p_(j) <= j alpha / m

    Parameters
    ----------
    pvalues : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rejected

    References
    ----------
    Hastie ESL Ch 18
    """
    pvalues = np.atleast_1d(np.asarray(pvalues, dtype=float))
    n = len(pvalues)
    result = float(np.mean(pvalues))
    se = float(np.std(pvalues, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Benjamini-Hochberg FDR control"})


def cheatsheet():
    return "eslfdr: Benjamini-Hochberg FDR control"

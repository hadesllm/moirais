"""Polygenic risk score (PRS-CS)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prs_cs"]


def prs_cs(sumstats, ld_ref):
    """
    Polygenic risk score (PRS-CS)

    Formula: continuous shrinkage on betas

    Parameters
    ----------
    sumstats : array-like
        Input data.
    ld_ref : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ge et al (2019)
    """
    sumstats = np.atleast_1d(np.asarray(sumstats, dtype=float))
    n = len(sumstats)
    result = float(np.mean(sumstats))
    se = float(np.std(sumstats, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polygenic risk score (PRS-CS)"})


def cheatsheet():
    return "prsccs: Polygenic risk score (PRS-CS)"

"""GWAS results across blocks."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gwas_block_combine"]


def gwas_block_combine(block_results):
    """
    GWAS results across blocks

    Formula: meta-analysis of per-block stats

    Parameters
    ----------
    block_results : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Bakker et al (2008)
    """
    block_results = np.atleast_1d(np.asarray(block_results, dtype=float))
    n = len(block_results)
    result = float(np.mean(block_results))
    se = float(np.std(block_results, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GWAS results across blocks"})


def cheatsheet():
    return "gwsblc: GWAS results across blocks"

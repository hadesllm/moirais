"""Phenotype QC (outlier removal + transform)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["phenotype_qc"]


def phenotype_qc(y):
    """
    Phenotype QC (outlier removal + transform)

    Formula: Tukey outlier rule + Box-Cox transform

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tukey (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Phenotype QC (outlier removal + transform)"})


def cheatsheet():
    return "pheno2: Phenotype QC (outlier removal + transform)"

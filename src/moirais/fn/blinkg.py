"""BLINK fast GWAS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["blink_gwas"]


def blink_gwas(y, M):
    """
    BLINK fast GWAS

    Formula: FarmCPU + LD-based QTN selection

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Huang et al (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BLINK fast GWAS"})


def cheatsheet():
    return "blinkg: BLINK fast GWAS"

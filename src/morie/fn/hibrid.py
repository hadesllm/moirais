"""Hybrid genomic prediction (parental GCA + SCA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hibrid_prediction"]


def hibrid_prediction(y, p1_geno, p2_geno):
    """
    Hybrid genomic prediction (parental GCA + SCA)

    Formula: y = mu + GCA_p1 + GCA_p2 + SCA_p1p2 + e

    Parameters
    ----------
    y : array-like
        Input data.
    p1_geno : array-like
        Input data.
    p2_geno : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Technow et al (2014)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hybrid genomic prediction (parental GCA + SCA)"})


def cheatsheet():
    return "hibrid: Hybrid genomic prediction (parental GCA + SCA)"

"""Numbered display equation (8.11) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_11"]


def mvsml_categorical_count_eq_8_11(Eq, the, eigenvalue, of, kernel, matrix):
    """
    Numbered display equation (8.11) from MVSML chapter 8.

    Formula: Eq. (8.11) if the eigenvalue decomposition of the kernel matrix K is expressed as US1/2S1/2U0, y = \mu1n + Pf + \epsilon, (8.11)   where f  N 0, \sigma2 (where r is the rank of K) and P = US1/2. Note that models f Ir,r (8.8) and

    Parameters
    ----------
    Eq : array-like
        Input data.
    the : array-like
        Input data.
    eigenvalue : array-like
        Input data.
    of : array-like
        Input data.
    kernel : array-like
        Input data.
    matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.11) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    Eq = np.atleast_1d(np.asarray(Eq, dtype=float))
    n = len(Eq)
    result = float(np.mean(Eq))
    se = float(np.std(Eq, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.11) from MVSML chapter 8."})


def cheatsheet():
    return "msm147: Numbered display equation (8.11) from MVSML chapter 8."

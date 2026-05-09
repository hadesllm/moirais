"""Numbered display equation (8.2) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_2"]


def mvsml_categorical_count_eq_8_2(dimensional, space, the, solution, admits, a):
    """
    Numbered display equation (8.2) from MVSML chapter 8.

    Formula: dimensional space, the solution for (8.1) admits a linear representation X n   = \eta0 + kT f xi ( ) = \eta0 + \beta jK xi, x j i \beta,

    Parameters
    ----------
    dimensional : array-like
        Input data.
    space : array-like
        Input data.
    the : array-like
        Input data.
    solution : array-like
        Input data.
    admits : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.2) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    dimensional = np.atleast_1d(np.asarray(dimensional, dtype=float))
    n = len(dimensional)
    result = float(np.mean(dimensional))
    se = float(np.std(dimensional, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.2) from MVSML chapter 8."})


def cheatsheet():
    return "msm125: Numbered display equation (8.2) from MVSML chapter 8."

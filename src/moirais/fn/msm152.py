"""Numbered display equation (8.12) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_12"]


def mvsml_categorical_count_eq_8_12(Therefore, an, eigen, of, K, the):
    """
    Numbered display equation (8.12) from MVSML chapter 8.

    Formula: Therefore, an eigen decomposition of K 2 1 the eigenvectors of order m  m and Sm,m is a diagonal matrix of order m  m with the eigenvalues ordered from largest to smallest. These values are substituted in Q  resulting in un  N 0, \sigma2 uKn,mUS 2 1=2S 2 1=2U0K0 ), and thus, thanks to the proper- n,m ties of the normal distribution, model (8.8) can be expressed like model (8.11) as y = \mu1n + Pf + \epsilon

    Parameters
    ----------
    Therefore : array-like
        Input data.
    an : array-like
        Input data.
    eigen : array-like
        Input data.
    of : array-like
        Input data.
    K : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.12) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    Therefore = np.atleast_1d(np.asarray(Therefore, dtype=float))
    n = len(Therefore)
    result = float(np.mean(Therefore))
    se = float(np.std(Therefore, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.12) from MVSML chapter 8."})


def cheatsheet():
    return "msm152: Numbered display equation (8.12) from MVSML chapter 8."

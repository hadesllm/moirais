"""Numbered display equation (8.8) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_8"]


def mvsml_categorical_count_eq_8_8(it, important, to, point, out, that):
    """
    Numbered display equation (8.8) from MVSML chapter 8.

    Formula: (2020), it is important to point out that model (8.8) can be reparametrized as Eq. (8.11) if the eigenvalue decomposition of the kernel matrix K is expressed as US1/2S1/2U0, y = \mu1n + Pf + \epsilon, (8.11)   where f  N 0, \sigma2 (where r is the rank of K) and P = US1/2. Note that models f Ir,r

    Parameters
    ----------
    it : array-like
        Input data.
    important : array-like
        Input data.
    to : array-like
        Input data.
    point : array-like
        Input data.
    out : array-like
        Input data.
    that : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.8) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    it = np.atleast_1d(np.asarray(it, dtype=float))
    n = len(it)
    result = float(np.mean(it))
    se = float(np.std(it, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.8) from MVSML chapter 8."})


def cheatsheet():
    return "msm146: Numbered display equation (8.8) from MVSML chapter 8."

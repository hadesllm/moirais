"""Numbered display equation (7.7) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_7"]


def mvsml_bayesian_regression_pt2_eq_7_7(l, When, p, large, n, direct):
    """
    Numbered display equation (7.7) from MVSML chapter 7.

    Formula: l=1 (7.8) When p is large ( p  n), direct optimization of ℓp(\beta; y) is almost impossible. An alternative is to use the sequential minimization optimization algorithm proposed by Zhu and Hastie (2004), which is applied after a transformation trick is used to make the involved computations feasible, because the number of parameters in the opti- mization is reduced to only (n + 1)C instead of ( p + 1)C. Another alternative available in the glmnet package is the one proposed by Friedman et al. (2010) that is similar to that of the logistic Ridge regression in Chap. 3. This consists of maximizing

    Parameters
    ----------
    l : array-like
        Input data.
    When : array-like
        Input data.
    p : array-like
        Input data.
    large : array-like
        Input data.
    n : array-like
        Input data.
    direct : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.7) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    l = np.atleast_1d(np.asarray(l, dtype=float))
    n = len(l)
    result = float(np.mean(l))
    se = float(np.std(l, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.7) from MVSML chapter 7."})


def cheatsheet():
    return "msm111: Numbered display equation (7.7) from MVSML chapter 7."

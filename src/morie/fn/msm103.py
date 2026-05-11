"""Numbered display equation (7.3) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_3"]


def mvsml_bayesian_regression_pt2_eq_7_3(L, XE, E, ZLg, e, The):
    """
    Numbered display equation (7.3) from MVSML chapter 7.

    Formula: (0.03) (0.1) (0.03) (0.1) L = XE\betaE + ZLg + e (7.5) The results are presented in Table 7.3 with the BS and PCCC metrics obtained in each partition of the random CV strategy. From this we can appreciate that the best performance with both metrics was obtained with the model that considered only the genetic effects (M3;

    Parameters
    ----------
    L : array-like
        Input data.
    XE : array-like
        Input data.
    E : array-like
        Input data.
    ZLg : array-like
        Input data.
    e : array-like
        Input data.
    The : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.3) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    L = np.atleast_1d(np.asarray(L, dtype=float))
    n = len(L)
    result = float(np.mean(L))
    se = float(np.std(L, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.3) from MVSML chapter 7."})


def cheatsheet():
    return "msm103: Numbered display equation (7.3) from MVSML chapter 7."

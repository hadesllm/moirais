"""Numbered display equation (9.7) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_7"]


def mvsml_ridge_lasso_elastic_eq_9_7(increasing, k, we, can, now, reformulate):
    """
    Numbered display equation (9.7) from MVSML chapter 9.

    Formula: 2 increasing for k\betak  0, we can now reformulate the optimization problem given in (9.6) as 346 9 Support Vector Machines and Support Vector Regression 1 k k2 minimize 2 \beta

    Parameters
    ----------
    increasing : array-like
        Input data.
    k : array-like
        Input data.
    we : array-like
        Input data.
    can : array-like
        Input data.
    now : array-like
        Input data.
    reformulate : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.7) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    increasing = np.atleast_1d(np.asarray(increasing, dtype=float))
    n = len(increasing)
    result = float(np.mean(increasing))
    se = float(np.std(increasing, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.7) from MVSML chapter 9."})


def cheatsheet():
    return "msm182: Numbered display equation (9.7) from MVSML chapter 9."

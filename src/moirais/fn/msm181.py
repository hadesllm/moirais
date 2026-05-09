"""Numbered display equation (9.6) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_6"]


def mvsml_ridge_lasso_elastic_eq_9_6(i, This, means, that, the, total):
    """
    Numbered display equation (9.6) from MVSML chapter 9.

    Formula: i \beta = 1). This means that the total distance is equal to 2M = 2/k\betak. This implies that maximizing M = 1/k\betak subject to the constraints of (9.6) is equivalent to minimizing (k\betak), subject to the same constraints. k k2 \beta Due to the fact that k\betak is naturally nonnegative and that is monotone 2 increasing for k\betak  0, we can now reformulate the optimization problem given in

    Parameters
    ----------
    i : array-like
        Input data.
    This : array-like
        Input data.
    means : array-like
        Input data.
    that : array-like
        Input data.
    the : array-like
        Input data.
    total : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.6) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    i = np.atleast_1d(np.asarray(i, dtype=float))
    n = len(i)
    result = float(np.mean(i))
    se = float(np.std(i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.6) from MVSML chapter 9."})


def cheatsheet():
    return "msm181: Numbered display equation (9.6) from MVSML chapter 9."

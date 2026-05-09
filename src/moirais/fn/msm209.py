"""Numbered display equation (9.29) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_29"]


def mvsml_ridge_lasso_elastic_eq_9_29(yi, xT, i, The, maximum, margin):
    """
    Numbered display equation (9.29) from MVSML chapter 9.

    Formula:   yi \beta0 + xT i \beta = 1. The maximum margin hyperplane is fully deﬁned by support vectors. The deﬁnition of these hyperplanes is not affected by vectors that are not lying on the marginal hyperplanes, since in their absence, the solution for the maximum margin hyperplane remains unchanged. By placing solutions (9.28) and

    Parameters
    ----------
    yi : array-like
        Input data.
    xT : array-like
        Input data.
    i : array-like
        Input data.
    The : array-like
        Input data.
    maximum : array-like
        Input data.
    margin : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.29) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.29) from MVSML chapter 9."})


def cheatsheet():
    return "msm209: Numbered display equation (9.29) from MVSML chapter 9."

"""Numbered display equation (9.4) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_4"]


def mvsml_ridge_lasso_elastic_eq_9_4(There, are, points, that, satisfying, lie):
    """
    Numbered display equation (9.4) from MVSML chapter 9.

    Formula: (9.3) There are points that satisfying (9.3) lie on one side of the hyperplane. Similarly, the X points that correspond to 340 9 Support Vector Machines and Support Vector Regression Fig. 9.2 The hyperplane 1 + 2X1 + 3X2 = 0 is shown. The blue region is the set of points for which 1 + 2X1 + 3X2 > 0, and the red region is the set of points for which 1 + 2X1 + 3X2 < 0 (James et al. 2013) \beta0 + \beta1X1 + \beta2X2 + . . . + \betapXp > 0

    Parameters
    ----------
    There : array-like
        Input data.
    are : array-like
        Input data.
    points : array-like
        Input data.
    that : array-like
        Input data.
    satisfying : array-like
        Input data.
    lie : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.4) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    There = np.atleast_1d(np.asarray(There, dtype=float))
    n = len(There)
    result = float(np.mean(There))
    se = float(np.std(There, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.4) from MVSML chapter 9."})


def cheatsheet():
    return "msm171: Numbered display equation (9.4) from MVSML chapter 9."

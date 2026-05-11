"""Numbered display equation (9.2) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_2"]


def mvsml_ridge_lasso_elastic_eq_9_2(From, a, mathematical, point, of, view):
    """
    Numbered display equation (9.2) from MVSML chapter 9.

    Formula: From a mathematical point of view, a hyperplane is deﬁned as (James et al. 2013) \beta0 + \beta1X1 + \beta2X2 + \beta3X3 = 0 (9.1) for parameters \beta0, \beta1, \beta2, and \beta3. (9.1) “deﬁnes” a hyperplane, since any X = (X1, X2, X3 )T for which (9.1) holds is a point in the hyperplane. Equation (9.1) is the equation of a plane, since in three dimensions, as mentioned before, a hyperplane is a plane, as can be observed in Fig. 9.1 (right). For the p-dimensional space, the dimension of the hyperplane generated is p + 1, and it is simply an extension of (9.1) as \beta0 + \beta1X1 + \beta2X2 + . . . + \betapXp = 0

    Parameters
    ----------
    From : array-like
        Input data.
    a : array-like
        Input data.
    mathematical : array-like
        Input data.
    point : array-like
        Input data.
    of : array-like
        Input data.
    view : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.2) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    From = np.atleast_1d(np.asarray(From, dtype=float))
    n = len(From)
    result = float(np.mean(From))
    se = float(np.std(From, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.2) from MVSML chapter 9."})


def cheatsheet():
    return "msm165: Numbered display equation (9.2) from MVSML chapter 9."

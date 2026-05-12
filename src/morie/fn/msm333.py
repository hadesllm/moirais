r"""Numbered display equation (3.1) from MVSML chapter 3.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_elements_lin_reg_eq_3_1"]


def mvsml_elements_lin_reg_eq_3_1(n, that, produces, an, equivalent, de):
    r"""
    Numbered display equation (3.1) from MVSML chapter 3.

    Formula: n that produces an equivalent deﬁnition to the penalized OLS presentation of the Ridge regression described before (Wakeﬁeld 2013; Hastie et al. 2009, 2015). This constrained reformulation gives a more transparent role than the one played by the tuning parameter, and among other things, suggests a convenient and common way of redeﬁning the Ridge estimator by standardizing the variables when these are of very different scales. A graphic representation of this constraint problem for \beta0 = 0 and p = 2 is given in Fig. 3.2, where the nested ellipsoids correspond to contour plots of RSS(\beta) and the green region is the restriction with t(\lambda) = 32, which contains the Ridge solution. The MLR deﬁned in

    Parameters
    ----------
    n : array-like
        Input data.
    that : array-like
        Input data.
    produces : array-like
        Input data.
    an : array-like
        Input data.
    equivalent : array-like
        Input data.
    de : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (3.1) [Multivariate Statistical Machine Learnin [Pages 71-108] [2026-04-16].pdf]
    r"""
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (3.1) from MVSML chapter 3."})


def cheatsheet():
    return "msm333: Numbered display equation (3.1) from MVSML chapter 3."

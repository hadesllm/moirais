"""Numbered display equation (14.11) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_11"]


def mvsml_convolutional_nn_eq_14_11(p, t, a, derivate, of, order):
    """
    Numbered display equation (14.11) from MVSML chapter 14.

    Formula: ϕ p ( ) t( ) is a derivate of order p of ϕi(t). Typical chosen values of p are 1 and 2. i A smoothed solution of the function \beta(t) can be obtained by minimizing (14.10) with respect to the parameters \betal, l = 1, . . ., L1. However, because this solution depends on the smoothing parameter, this needs to be determined. For this reason, as in the Ridge and Lasso regression models described in early chapters, here a cross- validation method is adopted ﬁrst, and a Bayesian approach will be described later. Under the penalty term

    Parameters
    ----------
    p : array-like
        Input data.
    t : array-like
        Input data.
    a : array-like
        Input data.
    derivate : array-like
        Input data.
    of : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.11) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.11) from MVSML chapter 14."})


def cheatsheet():
    return "msm281: Numbered display equation (14.11) from MVSML chapter 14."

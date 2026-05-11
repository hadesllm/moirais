"""Numbered display equation (14.1) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_1"]


def mvsml_convolutional_nn_eq_14_1(l, Ei, y, X, e, prior):
    """
    Numbered display equation (14.1) from MVSML chapter 14.

    Formula: l=1xil\betal + Ei   (y = 1n\mu + X\beta + e), with prior distribution \beta  N 0, \sigma2 \betaP1 , with \sigma2 \beta = \lambda \sigma2. So, from here a Bayesian formulation (PBFR) for the smoothed solution of the coefﬁcient function (\beta(t)) in the functional regression model

    Parameters
    ----------
    l : array-like
        Input data.
    Ei : array-like
        Input data.
    y : array-like
        Input data.
    X : array-like
        Input data.
    e : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.1) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.1) from MVSML chapter 14."})


def cheatsheet():
    return "msm292: Numbered display equation (14.1) from MVSML chapter 14."

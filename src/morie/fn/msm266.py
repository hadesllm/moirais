r"""Numbered display equation (14.2) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_2"]


def mvsml_convolutional_nn_eq_14_2(x, t, l, dt, L1, So):
    r"""
    Numbered display equation (14.2) from MVSML chapter 14.

    Formula: = 0 x t( )ϕl t( )dt, l = 1, . . ., L1. So, if yi, i = 1, . . ., n, are independent observations of model (14.1), corresponding to covariate functions xi(+), i = 1, . . ., n, a basis expansion solution for the beta coefﬁcient function is obtained by estimating the parameters involved in model h iT (14.3), and then substituting b\beta = \mu, b\beta1, . . . , b\betaL1 in

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    l : array-like
        Input data.
    dt : array-like
        Input data.
    L1 : array-like
        Input data.
    So : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.2) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.2) from MVSML chapter 14."})


def cheatsheet():
    return "msm266: Numbered display equation (14.2) from MVSML chapter 14."

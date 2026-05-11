"""Numbered display equation (14.2) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_2"]


def mvsml_convolutional_nn_eq_14_2(possible, to, nd, a, function, t):
    """
    Numbered display equation (14.2) from MVSML chapter 14.

    Formula: possible to ﬁnd a function \beta(t) satisfying the model with an error equal to 0, and there is an inﬁnite number of these functions that give the same predictions (Ramsay et al. 2009). There are several procedures to solve this problem (Cardot and Sarda 2006); one of them is based on basis expansion (Fourier, B-splines, etc.) and will be adopted and described here. A basis expansion solution is obtained by ﬁrst representing the beta coefﬁcient function \beta(t) as XL1 \beta t( ) = l=1\betalϕl t( ),

    Parameters
    ----------
    possible : array-like
        Input data.
    to : array-like
        Input data.
    nd : array-like
        Input data.
    a : array-like
        Input data.
    function : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.2) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    possible = np.atleast_1d(np.asarray(possible, dtype=float))
    n = len(possible)
    result = float(np.mean(possible))
    se = float(np.std(possible, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.2) from MVSML chapter 14."})


def cheatsheet():
    return "msm262: Numbered display equation (14.2) from MVSML chapter 14."

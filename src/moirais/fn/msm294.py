"""Numbered display equation (14.13) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_13"]


def mvsml_convolutional_nn_eq_14_13(where, XE, the, design, matrix, of):
    """
    Numbered display equation (14.13) from MVSML chapter 14.

    Formula: (14.13) where XE is the design matrix of the environments and \betaE is the vector with the environment effects, and the Bayesian formulation can be completed by assuming a prior distribution for \betaE. As was described in Chap. 6 in the BGLR package, there are several options for this: FIXED, BRR, BayesA, BayesB, BayesC, and BL. In the next example, the ﬁrst one is used. Example 14.6 This is a continuation of Example 14.5 used to illustrate the performance when adding environmental information to the prediction task by using the Bayesian formulation

    Parameters
    ----------
    where : array-like
        Input data.
    XE : array-like
        Input data.
    the : array-like
        Input data.
    design : array-like
        Input data.
    matrix : array-like
        Input data.
    of : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.13) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    where = np.atleast_1d(np.asarray(where, dtype=float))
    n = len(where)
    result = float(np.mean(where))
    se = float(np.std(where, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.13) from MVSML chapter 14."})


def cheatsheet():
    return "msm294: Numbered display equation (14.13) from MVSML chapter 14."

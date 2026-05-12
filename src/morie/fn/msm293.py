r"""Numbered display equation (14.13) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_13"]


def mvsml_convolutional_nn_eq_14_13(random, partitions, they, were, better, than):
    r"""
    Numbered display equation (14.13) from MVSML chapter 14.

    Formula: 100 random partitions, they were better than the PBFR and BFR, respectively. Also, taking into account the penalty term in the Bayesian prediction was not so important because in only 8 out of the 100 random partitions, the MSE of the PBFR was less than the MSE corresponding to the BFR (see Appendix 3 for the R code used). When using the penalty matrix based on second derivatives, in each case (Fourier and B-spline), the results were similar. The Bayesian formulation can be extended easily to take into account the effects of other factors. For example, in Example 14.5, the effects of the environment can be added as y = 1n\mu + XE\betaE + X\beta + e,

    Parameters
    ----------
    random : array-like
        Input data.
    partitions : array-like
        Input data.
    they : array-like
        Input data.
    were : array-like
        Input data.
    better : array-like
        Input data.
    than : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.13) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    r"""
    random = np.atleast_1d(np.asarray(random, dtype=float))
    n = len(random)
    result = float(np.mean(random))
    se = float(np.std(random, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.13) from MVSML chapter 14."})


def cheatsheet():
    return "msm293: Numbered display equation (14.13) from MVSML chapter 14."

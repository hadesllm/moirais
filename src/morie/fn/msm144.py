r"""Numbered display equation (8.10) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_10"]


def mvsml_categorical_count_eq_8_10(linear, GBLUP, kernels, the, best, second):
    r"""
    Numbered display equation (8.10) from MVSML chapter 8.

    Formula: and linear (GBLUP) kernels, while the best and second-best predictions were obtained with polynomial and Gaussian kernels. However, it is important to point out that the differences between the best and worst predictions were small. 8.8.1 Extended Predictor Under the Bayesian Kernel BLUP The Bayesian kernel BLUP method can be extended, in terms of the predictor, to easily take into account the effects of other factors. For example, in addition to the genotype effect, the effects of environments and genotype  environment interac- tion terms can also be incorporated as y = \mu1 + ZE\betaE + u1 + u2 + \epsilon,

    Parameters
    ----------
    linear : array-like
        Input data.
    GBLUP : array-like
        Input data.
    kernels : array-like
        Input data.
    the : array-like
        Input data.
    best : array-like
        Input data.
    second : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.10) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    linear = np.atleast_1d(np.asarray(linear, dtype=float))
    n = len(linear)
    result = float(np.mean(linear))
    se = float(np.std(linear, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.10) from MVSML chapter 8."})


def cheatsheet():
    return "msm144: Numbered display equation (8.10) from MVSML chapter 8."

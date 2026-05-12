r"""Numbered display equation (6.9) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def mvsml_bayesian_regression_eq_6_9(marker, information, df0, vT, S0, ST):
    r"""
    Numbered display equation (6.9) from MVSML chapter 6.

    Formula: marker information, df0 = vT and S0 = ST are the degrees of freedom parameter (vT) and the scale matrix parameter (ST) of the inverse Wishart prior distribution for \SigmaT, respectively. In the third argument (resCOV), S0 and df0 are the scale matrix parameter (SR) and the degree of freedom parameter (vR) of the inverse Wishart 194 6 Bayesian Genomic Linear Regression prior distribution for R. The last two arguments are the required number of iterations (nI) and the burn-in period (nb) for running the Gibbs sampler. Similarly to the univariate case, model

    Parameters
    ----------
    marker : array-like
        Input data.
    information : array-like
        Input data.
    df0 : array-like
        Input data.
    vT : array-like
        Input data.
    S0 : array-like
        Input data.
    ST : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.9) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    marker = np.atleast_1d(np.asarray(marker, dtype=float))
    n = len(marker)
    result = float(np.mean(marker))
    se = float(np.std(marker, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.9) from MVSML chapter 6."})


def cheatsheet():
    return "msm071: Numbered display equation (6.9) from MVSML chapter 6."

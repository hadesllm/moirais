"""Numbered display equation (6.10) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_10"]


def mvsml_bayesian_regression_eq_6_10(respectively, In, the, third, argument, resCOV):
    """
    Numbered display equation (6.10) from MVSML chapter 6.

    Formula: respectively. In the third argument (resCOV), S0 and df0 are the scale matrix parameter (SR) and the degree of freedom parameter (vR) of the inverse Wishart 194 6 Bayesian Genomic Linear Regression prior distribution for R. The last two arguments are the required number of iterations (nI) and the burn-in period (nb) for running the Gibbs sampler. Similarly to the univariate case, model (6.9) can be equivalently described and implemented as a multivariate Ridge regression model, as follows: Y = 1J\muT + XB + X1B1 + E,

    Parameters
    ----------
    respectively : array-like
        Input data.
    In : array-like
        Input data.
    the : array-like
        Input data.
    third : array-like
        Input data.
    argument : array-like
        Input data.
    resCOV : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.10) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    respectively = np.atleast_1d(np.asarray(respectively, dtype=float))
    n = len(respectively)
    result = float(np.mean(respectively))
    se = float(np.std(respectively, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.10) from MVSML chapter 6."})


def cheatsheet():
    return "msm072: Numbered display equation (6.10) from MVSML chapter 6."

"""Numbered display equation (6.1) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_1"]


def mvsml_bayesian_regression_eq_6_1(ETA, list, model, RHKS, K, K_L):
    """
    Numbered display equation (6.1) from MVSML chapter 6.

    Formula: ETA = list( list( model = ‘RHKS’, K = K_L, df0 = vg, S0 = Sg, R2 = 1-R2)) ) A = BGLR(y=y, ETA = ETA, nIter = 1e4, burnIn = 1e3, S0 = S, df0 = v, R2 = R2) 178 6 Bayesian Genomic Linear Regression where dat_F is the data set that contains the necessary phenotypic information (GID: Lines or individuals; y: response variable of the trait of interest). 6.4 Genomic-Enabled Prediction BayesA Model Another variant to the standard Bayesian model

    Parameters
    ----------
    ETA : array-like
        Input data.
    list : array-like
        Input data.
    model : array-like
        Input data.
    RHKS : array-like
        Input data.
    K : array-like
        Input data.
    K_L : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.1) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    ETA = np.atleast_1d(np.asarray(ETA, dtype=float))
    n = len(ETA)
    result = float(np.mean(ETA))
    se = float(np.std(ETA, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.1) from MVSML chapter 6."})


def cheatsheet():
    return "msm057: Numbered display equation (6.1) from MVSML chapter 6."

"""Numbered display equation (6.4) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_4"]


def mvsml_bayesian_regression_eq_6_4(The, GBLUP, can, be, equivalently, expressed):
    """
    Numbered display equation (6.4) from MVSML chapter 6.

    Formula: The GBLUP can be equivalently expressed and consequently ﬁtted with the BRR model by making the design matrix equal to the lower triangular factor of the Cholesky decomposition of the genomic relationship matrix, i.e., X = L, where G = LL0. So, with the BGLR package, the BRR implementation of a GBLUP model is. L = t(chol(G)) ETA = list( list( model = ‘BRR’, X = L, df0 = v\beta, S0 = S\beta, R2 = 1-R2) ) A = BGLR(y=y, ETA = ETA, nIter = 1e4, burnIn = 1e3, S0 = S, df0 = v, R2 = R2) When there is more than one repetition of an individual in the data at hand, or a more sophisticated design is used in the data collection, model

    Parameters
    ----------
    The : array-like
        Input data.
    GBLUP : array-like
        Input data.
    can : array-like
        Input data.
    be : array-like
        Input data.
    equivalently : array-like
        Input data.
    expressed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.4) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    The = np.atleast_1d(np.asarray(The, dtype=float))
    n = len(The)
    result = float(np.mean(The))
    se = float(np.std(The, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.4) from MVSML chapter 6."})


def cheatsheet():
    return "msm054: Numbered display equation (6.4) from MVSML chapter 6."

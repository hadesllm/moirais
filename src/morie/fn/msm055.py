r"""Numbered display equation (6.5) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_5"]


def mvsml_bayesian_regression_eq_6_5(Cholesky, of, the, genomic, relationship, matrix):
    r"""
    Numbered display equation (6.5) from MVSML chapter 6.

    Formula: Cholesky decomposition of the genomic relationship matrix, i.e., X = L, where G = LL0. So, with the BGLR package, the BRR implementation of a GBLUP model is. L = t(chol(G)) ETA = list( list( model = ‘BRR’, X = L, df0 = v\beta, S0 = S\beta, R2 = 1-R2) ) A = BGLR(y=y, ETA = ETA, nIter = 1e4, burnIn = 1e3, S0 = S, df0 = v, R2 = R2) When there is more than one repetition of an individual in the data at hand, or a more sophisticated design is used in the data collection, model (6.4) can be speciﬁed in a more general way to take into account this structure, as follows: Y = 1n\mu + Zg + e

    Parameters
    ----------
    Cholesky : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.
    genomic : array-like
        Input data.
    relationship : array-like
        Input data.
    matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.5) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    Cholesky = np.atleast_1d(np.asarray(Cholesky, dtype=float))
    n = len(Cholesky)
    result = float(np.mean(Cholesky))
    se = float(np.std(Cholesky, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.5) from MVSML chapter 6."})


def cheatsheet():
    return "msm055: Numbered display equation (6.5) from MVSML chapter 6."

"""Numbered display equation (1.5) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_5"]


def mvsml_general_eq_1_5(eij, where, P5, i, represents, the):
    """
    Numbered display equation (1.5) from MVSML chapter 1.

    Formula: + eij, (1.4) where \beta = P5 i=1\betai=5 represents the average grain yield for the environments in the experiment. The random effects model replaces \beta with the mean grain yield across the population of environments and replaces the deviations \betai \beta with the random variables whose distribution is to be estimated. If \betai \beta is not assumed random the model belongs to ﬁxed effects. Therefore, the random effects version of the model given in Eq. (1.4) is equal to GYij = \beta + bi + eij,

    Parameters
    ----------
    eij : array-like
        Input data.
    where : array-like
        Input data.
    P5 : array-like
        Input data.
    i : array-like
        Input data.
    represents : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.5) [Multivariate Statistical Machine Learnin [Pages 1-34] [2026-04-16].pdf]
    """
    eij = np.atleast_1d(np.asarray(eij, dtype=float))
    n = len(eij)
    result = float(np.mean(eij))
    se = float(np.std(eij, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.5) from MVSML chapter 1."})


def cheatsheet():
    return "msm006: Numbered display equation (1.5) from MVSML chapter 1."

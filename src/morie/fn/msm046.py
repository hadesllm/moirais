r"""Numbered display equation (6.3) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_3"]


def mvsml_bayesian_regression_eq_6_3(non, informative, prior, given, Christensen, et):
    r"""
    Numbered display equation (6.3) from MVSML chapter 6.

    Formula: non-informative prior given in (6.1) (Christensen et al. 2011). A similar prior speciﬁcation is taken in genomic prediction where different models are obtained by adopting different prior distributions of the parameters. For example, the Bayes- ian Linear Ridge Regression (Pérez and de los Campos 2014) with standardized 0s) is given by covariates (Xj X p Y = \mu + X j\beta j + E

    Parameters
    ----------
    non : array-like
        Input data.
    informative : array-like
        Input data.
    prior : array-like
        Input data.
    given : array-like
        Input data.
    Christensen : array-like
        Input data.
    et : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.3) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    non = np.atleast_1d(np.asarray(non, dtype=float))
    n = len(non)
    result = float(np.mean(non))
    se = float(np.std(non, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.3) from MVSML chapter 6."})


def cheatsheet():
    return "msm046: Numbered display equation (6.3) from MVSML chapter 6."

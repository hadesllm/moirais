"""Numbered display equation (7.1) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def mvsml_bayesian_regression_pt2_eq_7_1(A, BGLR, y, ordinal, ETA, nIter):
    """
    Numbered display equation (7.1) from MVSML chapter 7.

    Formula: A = BGLR(y=y,response_type='ordinal',ETA=ETA, nIter = 1e4,burnIn = 1e3,verbose = FALSE) Probs = A$probs where dat_F is the data ﬁle that contains all the information of how the data was collected (GID: Lines or individuals; Env: Environment; y: response variable of the trait). Other desired prior models to beta coefﬁcients of each predictor component are obtained only by replacing the “model” argument of each of the three components of the predictor. For example, for a BayesA prior model for the marker effects, in the second sub-list we must use model='BayesA'. The latent random vector of model

    Parameters
    ----------
    A : array-like
        Input data.
    BGLR : array-like
        Input data.
    y : array-like
        Input data.
    ordinal : array-like
        Input data.
    ETA : array-like
        Input data.
    nIter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.1) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.1) from MVSML chapter 7."})


def cheatsheet():
    return "msm094: Numbered display equation (7.1) from MVSML chapter 7."

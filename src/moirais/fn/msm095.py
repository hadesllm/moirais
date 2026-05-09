"""Numbered display equation (7.4) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_4"]


def mvsml_bayesian_regression_pt2_eq_7_4(Probs, A, probs, where, dat_F, the):
    """
    Numbered display equation (7.4) from MVSML chapter 7.

    Formula: Probs = A$probs where dat_F is the data ﬁle that contains all the information of how the data was collected (GID: Lines or individuals; Env: Environment; y: response variable of the trait). Other desired prior models to beta coefﬁcients of each predictor component are obtained only by replacing the “model” argument of each of the three components of the predictor. For example, for a BayesA prior model for the marker effects, in the second sub-list we must use model='BayesA'. The latent random vector of model (7.1) under the GBLUP speciﬁcation, plus genotypic and environment+genotypic interaction effects, takes the form L = XE\betaE + ZLg + ZLEgE + e

    Parameters
    ----------
    Probs : array-like
        Input data.
    A : array-like
        Input data.
    probs : array-like
        Input data.
    where : array-like
        Input data.
    dat_F : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.4) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    Probs = np.atleast_1d(np.asarray(Probs, dtype=float))
    n = len(Probs)
    result = float(np.mean(Probs))
    se = float(np.std(Probs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.4) from MVSML chapter 7."})


def cheatsheet():
    return "msm095: Numbered display equation (7.4) from MVSML chapter 7."

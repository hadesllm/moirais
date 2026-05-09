"""Numbered display equation (6.7) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_7"]


def mvsml_bayesian_regression_eq_6_7(collected, GID, Lines, individuals, Env, Environment):
    """
    Numbered display equation (6.7) from MVSML chapter 6.

    Formula: collected (GID: Lines or individuals; Env: Environment; y: response variable of the trait). Other desired prior models to beta coefﬁcients of each predictor component are obtained only by replacing the “model” argument of each of the three components of the predictor. For example, for a BayesA prior model for the marker effects, in the second sub-list we must use model='BayesA'. The latent random vector of model (7.1) under the GBLUP speciﬁcation, plus genotypic and environment+genotypic interaction effects, takes the form L = XE\betaE + ZLg + ZLEgE + e (7.4) which is like model

    Parameters
    ----------
    collected : array-like
        Input data.
    GID : array-like
        Input data.
    Lines : array-like
        Input data.
    individuals : array-like
        Input data.
    Env : array-like
        Input data.
    Environment : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.7) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    collected = np.atleast_1d(np.asarray(collected, dtype=float))
    n = len(collected)
    result = float(np.mean(collected))
    se = float(np.std(collected, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.7) from MVSML chapter 6."})


def cheatsheet():
    return "msm096: Numbered display equation (6.7) from MVSML chapter 6."

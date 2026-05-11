"""Numbered display equation (7.1) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def mvsml_bayesian_regression_pt2_eq_7_1(the, PCCC, of, second, model, was):
    """
    Numbered display equation (7.1) from MVSML chapter 7.

    Formula: the PCCC of the second model was 0.50% greater than that of the ﬁrst. The R code to reproduce the results is in Appendix 3. 7.3 Ordinal Logistic Regression As described at the beginning of this chapter, the ordinal logistic model is given in model (7.1) but with F the cumulative logistic distribution. Again, as in the ordinal probit regression model, the posterior distribution of the parameter is not easy to simulate and numerical methods are required. Here we describe the Gibbs sampler proposed by Montesinos-López et al. (2015b), which in addition to the latent variable Li in the representation of model

    Parameters
    ----------
    the : array-like
        Input data.
    PCCC : array-like
        Input data.
    of : array-like
        Input data.
    second : array-like
        Input data.
    model : array-like
        Input data.
    was : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.1) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    the = np.atleast_1d(np.asarray(the, dtype=float))
    n = len(the)
    result = float(np.mean(the))
    se = float(np.std(the, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.1) from MVSML chapter 7."})


def cheatsheet():
    return "msm104: Numbered display equation (7.1) from MVSML chapter 7."

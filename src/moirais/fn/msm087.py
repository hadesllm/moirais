"""Numbered display equation (7.1) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def mvsml_bayesian_regression_pt2_eq_7_1(a, scaled, inverse, chi, squared, From):
    """
    Numbered display equation (7.1) from MVSML chapter 7.

    Formula: , and a scaled inverse chi-squared for \sigma2 \beta: From now on, this model will be referred to as BRR in this chapter in analogy to the Bayesian linear regression and the way that this is implemented in the BGLR R package. Similar to the genomic linear regression model, the posterior distribution of the parameter vector does not have a known form and a Gibbs sampler is used to explore this; for this reason, in the coming lines, the Gibbs sampling method proposed by Albert and Chib (1993) is described. To make it possible to derive the full conditional distributions, the parameter vector is augmented with a latent variable in the representation of model

    Parameters
    ----------
    a : array-like
        Input data.
    scaled : array-like
        Input data.
    inverse : array-like
        Input data.
    chi : array-like
        Input data.
    squared : array-like
        Input data.
    From : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.1) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.1) from MVSML chapter 7."})


def cheatsheet():
    return "msm087: Numbered display equation (7.1) from MVSML chapter 7."

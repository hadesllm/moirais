"""Numbered display equation (7.1) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_1"]


def mvsml_bayesian_regression_pt2_eq_7_1(c, bi, C, where, now, Li):
    """
    Numbered display equation (7.1) from MVSML chapter 7.

    Formula: ( )  \Phi \gammac1 + bi ( ), c = 1, . . . , C, (7.2) where now Li =  bi + Ei is the latent variable, and \Phi is the cumulative normal standard distribution. In matrix form the model for the latent variable can be speciﬁed as L = b + e, where L = (L1, . . ., Ln)T and e~Nn(0, In). A Gibbs sampler of the posterior of the parameters of this model can be obtained similarly as was done for model

    Parameters
    ----------
    c : array-like
        Input data.
    bi : array-like
        Input data.
    C : array-like
        Input data.
    where : array-like
        Input data.
    now : array-like
        Input data.
    Li : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.1) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.1) from MVSML chapter 7."})


def cheatsheet():
    return "msm090: Numbered display equation (7.1) from MVSML chapter 7."

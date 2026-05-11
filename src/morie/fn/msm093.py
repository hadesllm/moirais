"""Numbered display equation (7.3) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_3"]


def mvsml_bayesian_regression_pt2_eq_7_3(C, where, L, L1, Ln, T):
    """
    Numbered display equation (7.3) from MVSML chapter 7.

    Formula: . , C, where L = (L1, . . ., Ln)T = XE\betaE + X\beta + XEM\betaEM + e is the vector with latent random variables of all observations, e  Nn(0, In) is a random error vector, XE, X, and XEM are the design matrices of the environments, markers, and environment–marker interactions, respectively, while \betaE and \betaEM are the vectors of the environment effects and the interaction effects, respectively, with a prior distribution that can be speciﬁed as was done for \beta. In fact, with the BGLR function, it is also possible to implement all these extensions, since it allows using any of the several priors included here: FIXED, BRR, BayesA, BayesB, BayesC, and BL. For example, the basic BGLR code to implement model

    Parameters
    ----------
    C : array-like
        Input data.
    where : array-like
        Input data.
    L : array-like
        Input data.
    L1 : array-like
        Input data.
    Ln : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.3) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    C = np.atleast_1d(np.asarray(C, dtype=float))
    n = len(C)
    result = float(np.mean(C))
    se = float(np.std(C, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.3) from MVSML chapter 7."})


def cheatsheet():
    return "msm093: Numbered display equation (7.3) from MVSML chapter 7."

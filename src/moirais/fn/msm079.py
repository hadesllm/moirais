"""Numbered display equation (6.11) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_11"]


def mvsml_bayesian_regression_eq_6_11(G, which, means, that, E, j):
    r"""
    Numbered display equation (6.11) from MVSML chapter 6.

    Formula:   G-1 \Sigma-1 which means that \SigmaE j -  IW \upsilonE + JL, bT b 2 + SE . 2 T A Gibbs sampler to explore the joint posterior distribution of parameters of model

    Parameters
    ----------
    G : array-like
        Input data.
    which : array-like
        Input data.
    means : array-like
        Input data.
    that : array-like
        Input data.
    E : array-like
        Input data.
    j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.11) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.11) from MVSML chapter 6."})


def cheatsheet():
    return "msm079: Numbered display equation (6.11) from MVSML chapter 6."

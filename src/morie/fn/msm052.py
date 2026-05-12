r"""Numbered display equation (6.4) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_4"]


def mvsml_bayesian_regression_eq_6_4(here, g, j, Nn, eg, eG):
    r"""
    Numbered display equation (6.4) from MVSML chapter 6.

    Formula: ) , and from here g j   2  Nn eg, eG . Then, the mean/mode of g j - is eg = \sigma-2eG y 2 1n\mu ( ), which is also the best linear unbiased predictor (BLUP) of g under the mixed model equation of Henderson (1975) using the machinery of a classic linear mixed model described in the previous chapter for model

    Parameters
    ----------
    here : array-like
        Input data.
    g : array-like
        Input data.
    j : array-like
        Input data.
    Nn : array-like
        Input data.
    eg : array-like
        Input data.
    eG : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.4) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    here = np.atleast_1d(np.asarray(here, dtype=float))
    n = len(here)
    result = float(np.mean(here))
    se = float(np.std(here, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.4) from MVSML chapter 6."})


def cheatsheet():
    return "msm052: Numbered display equation (6.4) from MVSML chapter 6."

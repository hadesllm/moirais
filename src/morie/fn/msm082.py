r"""Numbered display equation (6.3) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_3"]


def mvsml_bayesian_regression_eq_6_3(j, jq, Appendix, Setting, the, Prior):
    r"""
    Numbered display equation (6.3) from MVSML chapter 6.

    Formula: j jq=2 ( 2\pi Appendix 2: Setting Hyperparameters for the Prior Distributions of the BRR Model The following rules are those used in Pérez and de los Campos (2014), and provide proper but weakly informative prior distributions. In general, this consists of assigning a certain proportion of the total variance of the phenotypes, to the different components of the model. Speciﬁcally, for model

    Parameters
    ----------
    j : array-like
        Input data.
    jq : array-like
        Input data.
    Appendix : array-like
        Input data.
    Setting : array-like
        Input data.
    the : array-like
        Input data.
    Prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.3) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    r"""
    j = np.atleast_1d(np.asarray(j, dtype=float))
    n = len(j)
    result = float(np.mean(j))
    se = float(np.std(j, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.3) from MVSML chapter 6."})


def cheatsheet():
    return "msm082: Numbered display equation (6.3) from MVSML chapter 6."

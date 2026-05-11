"""Numbered display equation (6.9) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_9"]


def mvsml_bayesian_regression_eq_6_9(Bayesian, Genomic, Multi, trait, environment, Model):
    """
    Numbered display equation (6.9) from MVSML chapter 6.

    Formula: 6.9 Bayesian Genomic Multi-trait and Multi-environment Model (BMTME) 195 MSE in the ﬁrst trait is mainly because the measurement scale is greater than in the second trait. The R codes to reproduce these results (Table 6.4) are shown in Appendix 5. 6.9 Bayesian Genomic Multi-trait and Multi-environment Model (BMTME) Model

    Parameters
    ----------
    Bayesian : array-like
        Input data.
    Genomic : array-like
        Input data.
    Multi : array-like
        Input data.
    trait : array-like
        Input data.
    environment : array-like
        Input data.
    Model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.9) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    Bayesian = np.atleast_1d(np.asarray(Bayesian, dtype=float))
    n = len(Bayesian)
    result = float(np.mean(Bayesian))
    se = float(np.std(Bayesian, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.9) from MVSML chapter 6."})


def cheatsheet():
    return "msm075: Numbered display equation (6.9) from MVSML chapter 6."

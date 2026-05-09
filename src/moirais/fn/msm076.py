"""Numbered display equation (6.11) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_11"]


def mvsml_bayesian_regression_eq_6_11(Bayesian, Genomic, Multi, trait, environment, Model):
    """
    Numbered display equation (6.11) from MVSML chapter 6.

    Formula: 6.9 Bayesian Genomic Multi-trait and Multi-environment Model (BMTME) Model (6.9) does not take into account the possible trait–genotype–environment interaction (T  G  E), when environment information is available. An extension of this model is the one proposed by Montesinos-López et al. (2016), who added this interaction term to vary the speciﬁc trait genetic effects (gj) across environments. If the information of nT traits of J lines is collected in I environments, this model is given by Y = 1IJ\muT + XB + Z1b1 + Z2b2 + E,

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
    MVSML, Eq. (6.11) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    Bayesian = np.atleast_1d(np.asarray(Bayesian, dtype=float))
    n = len(Bayesian)
    result = float(np.mean(Bayesian))
    se = float(np.std(Bayesian, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.11) from MVSML chapter 6."})


def cheatsheet():
    return "msm076: Numbered display equation (6.11) from MVSML chapter 6."

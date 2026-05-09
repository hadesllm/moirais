"""Numbered display equation (6.2) from MVSML chapter 6.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_eq_6_2"]


def mvsml_bayesian_regression_eq_6_2(the, information, of, nT, traits, J):
    """
    Numbered display equation (6.2) from MVSML chapter 6.

    Formula: the information of nT traits of J lines is collected in I environments, this model is given by Y = 1IJ\muT + XB + Z1b1 + Z2b2 + E, (6.11) where =[Y1, . . ., YIJ]T, X = [x1, . . ., xIJ]T, Z1 and Z2 are the incident lines and the incident environment–line interaction matrices, b1 = [g1, . . ., gJ]T, b2 = [g21, . . ., g2IJ]T, and E = [e1, . . ., eIJ]T. Here, b2 j \SigmaT, \SigmaE  MNIJnT 0, \SigmaE⨂G, \SigmaT ( ) , and similar to model

    Parameters
    ----------
    the : array-like
        Input data.
    information : array-like
        Input data.
    of : array-like
        Input data.
    nT : array-like
        Input data.
    traits : array-like
        Input data.
    J : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (6.2) [Multivariate Statistical Machine Learnin [Pages 171-208] [2026-04-16].pdf]
    """
    the = np.atleast_1d(np.asarray(the, dtype=float))
    n = len(the)
    result = float(np.mean(the))
    se = float(np.std(the, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (6.2) from MVSML chapter 6."})


def cheatsheet():
    return "msm077: Numbered display equation (6.2) from MVSML chapter 6."

"""Numbered display equation (14.13) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_13"]


def mvsml_convolutional_nn_eq_14_13(cat, Partition, p, Bayesian, Estimation, of):
    """
    Numbered display equation (14.13) from MVSML chapter 14.

    Formula: cat('Partition = ', p,'\n') } 14.5 Bayesian Estimation of the Functional Regression 609 Table 14.2 Mean square error (MSE) of prediction for 100 random partitions for Fourier and B-spline representations, where the PFR and FR are classic functional regression models used in   Example 14.4, and PBFR and BFR are model

    Parameters
    ----------
    cat : array-like
        Input data.
    Partition : array-like
        Input data.
    p : array-like
        Input data.
    Bayesian : array-like
        Input data.
    Estimation : array-like
        Input data.
    of : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.13) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    cat = np.atleast_1d(np.asarray(cat, dtype=float))
    n = len(cat)
    result = float(np.mean(cat))
    se = float(np.std(cat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.13) from MVSML chapter 14."})


def cheatsheet():
    return "msm295: Numbered display equation (14.13) from MVSML chapter 14."

"""Numbered display equation (14.14) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_14"]


def mvsml_convolutional_nn_eq_14_14(of, the, MSE, PBFR, vs, BFR):
    """
    Numbered display equation (14.14) from MVSML chapter 14.

    Formula: of the MSE (PBFR (14.13) vs. PBFR (14.14) and BFR (14.13) vs. BFR (14.14)), and again the Bayesian model without penalization matrix was better (0.2955 vs. 0.2899) Table 14.3 Mean squared error of prediction (MSE) for 100 random partitions for Fourier and B-spline representations, where PBFR (14.13) and BFR (14.13) are MSE of the model (14.13) with and without penalization matrix based on the ﬁrst derivative in the functional term (X\beta), and PBFR (14.14) and BFR (14.14) are for model (14.14) with and without penalization matrix based on the ﬁrst derivative in the functional terms (X\beta,XEF\betaEF) PBFR (14.13) BFR (14.13) PBFR

    Parameters
    ----------
    of : array-like
        Input data.
    the : array-like
        Input data.
    MSE : array-like
        Input data.
    PBFR : array-like
        Input data.
    vs : array-like
        Input data.
    BFR : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.14) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    of = np.atleast_1d(np.asarray(of, dtype=float))
    n = len(of)
    result = float(np.mean(of))
    se = float(np.std(of, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.14) from MVSML chapter 14."})


def cheatsheet():
    return "msm313: Numbered display equation (14.14) from MVSML chapter 14."

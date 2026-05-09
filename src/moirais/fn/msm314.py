"""Numbered display equation (14.14) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_14"]


def mvsml_convolutional_nn_eq_14_14(again, the, Bayesian, model, without, penalization):
    """
    Numbered display equation (14.14) from MVSML chapter 14.

    Formula: again the Bayesian model without penalization matrix was better (0.2955 vs. 0.2899) Table 14.3 Mean squared error of prediction (MSE) for 100 random partitions for Fourier and B-spline representations, where PBFR (14.13) and BFR (14.13) are MSE of the model (14.13) with and without penalization matrix based on the ﬁrst derivative in the functional term (X\beta), and PBFR (14.14) and BFR (14.14) are for model (14.14) with and without penalization matrix based on the ﬁrst derivative in the functional terms (X\beta,XEF\betaEF) PBFR (14.13) BFR (14.13) PBFR (14.14) BFR

    Parameters
    ----------
    again : array-like
        Input data.
    the : array-like
        Input data.
    Bayesian : array-like
        Input data.
    model : array-like
        Input data.
    without : array-like
        Input data.
    penalization : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.14) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    again = np.atleast_1d(np.asarray(again, dtype=float))
    n = len(again)
    result = float(np.mean(again))
    se = float(np.std(again, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.14) from MVSML chapter 14."})


def cheatsheet():
    return "msm314: Numbered display equation (14.14) from MVSML chapter 14."

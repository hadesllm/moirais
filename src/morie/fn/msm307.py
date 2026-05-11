"""Numbered display equation (14.13) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_13"]


def mvsml_convolutional_nn_eq_14_13(nd, without, BFR, rst, derivative, penalization):
    """
    Numbered display equation (14.13) from MVSML chapter 14.

    Formula: nd without (BFR (14.14)) ﬁrst derivative penalization. The results are shown in Table 14.3, together with the prediction performance of the model with no interaction term (model 14.13). We can observe that by adding the interaction of environment with the functional covariate, both Bayesian models (PBFR and BFR) resulted in a reduction on average of about 35% of the MSE (PBFR (14.13) vs. PBFR (14.14) and BFR (14.13) vs. BFR (14.14)), and again the Bayesian model without penalization matrix was better (0.2955 vs. 0.2899) Table 14.3 Mean squared error of prediction (MSE) for 100 random partitions for Fourier and B-spline representations, where PBFR (14.13) and BFR (14.13) are MSE of the model

    Parameters
    ----------
    nd : array-like
        Input data.
    without : array-like
        Input data.
    BFR : array-like
        Input data.
    rst : array-like
        Input data.
    derivative : array-like
        Input data.
    penalization : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.13) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    nd = np.atleast_1d(np.asarray(nd, dtype=float))
    n = len(nd)
    result = float(np.mean(nd))
    se = float(np.std(nd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.13) from MVSML chapter 14."})


def cheatsheet():
    return "msm307: Numbered display equation (14.13) from MVSML chapter 14."

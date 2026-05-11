"""Numbered display equation (14.14) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_14"]


def mvsml_convolutional_nn_eq_14_14(ance, of, the, model, no, interaction):
    """
    Numbered display equation (14.14) from MVSML chapter 14.

    Formula: ance of the model with no interaction term (model 14.13). We can observe that by adding the interaction of environment with the functional covariate, both Bayesian models (PBFR and BFR) resulted in a reduction on average of about 35% of the MSE (PBFR (14.13) vs. PBFR (14.14) and BFR (14.13) vs. BFR (14.14)), and again the Bayesian model without penalization matrix was better (0.2955 vs. 0.2899) Table 14.3 Mean squared error of prediction (MSE) for 100 random partitions for Fourier and B-spline representations, where PBFR (14.13) and BFR (14.13) are MSE of the model (14.13) with and without penalization matrix based on the ﬁrst derivative in the functional term (X\beta), and PBFR (14.14) and BFR

    Parameters
    ----------
    ance : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.
    model : array-like
        Input data.
    no : array-like
        Input data.
    interaction : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.14) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    ance = np.atleast_1d(np.asarray(ance, dtype=float))
    n = len(ance)
    result = float(np.mean(ance))
    se = float(np.std(ance, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.14) from MVSML chapter 14."})


def cheatsheet():
    return "msm309: Numbered display equation (14.14) from MVSML chapter 14."

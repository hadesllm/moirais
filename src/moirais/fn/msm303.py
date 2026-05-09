"""Numbered display equation (14.13) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_13"]


def mvsml_convolutional_nn_eq_14_13(xT, n, This, model, was, also):
    """
    Numbered display equation (14.13) from MVSML chapter 14.

    Formula: 0 xT n This model was also evaluated with 100 random partitions with both Fourier and B-spline basis and with (PBFR (14.14)) and without (BFR (14.14)) ﬁrst derivative penalization. The results are shown in Table 14.3, together with the prediction performance of the model with no interaction term (model 14.13). We can observe that by adding the interaction of environment with the functional covariate, both Bayesian models (PBFR and BFR) resulted in a reduction on average of about 35% of the MSE (PBFR (14.13) vs. PBFR (14.14) and BFR

    Parameters
    ----------
    xT : array-like
        Input data.
    n : array-like
        Input data.
    This : array-like
        Input data.
    model : array-like
        Input data.
    was : array-like
        Input data.
    also : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.13) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    xT = np.atleast_1d(np.asarray(xT, dtype=float))
    n = len(xT)
    result = float(np.mean(xT))
    se = float(np.std(xT, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.13) from MVSML chapter 14."})


def cheatsheet():
    return "msm303: Numbered display equation (14.13) from MVSML chapter 14."

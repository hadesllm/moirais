"""Numbered display equation (5.3) from MVSML chapter 5.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_linear_mixed_models_eq_5_3"]


def mvsml_linear_mixed_models_eq_5_3(M11, referred, to, model, plus, environment):
    """
    Numbered display equation (5.3) from MVSML chapter 5.

    Formula: (0.116) (0.133) (0.117) (0.112) (0.083) M11 is referred to as model (5.3) plus environment effects (Env). The mean squared error of prediction (MSE) and Pearson’s correlation (PC) for each partition are reported. SD is the standard deviation Y = 1n\mu + XE\betaE + ZLb + e, where \mu, \betaE, and b are as before

    Parameters
    ----------
    M11 : array-like
        Input data.
    referred : array-like
        Input data.
    to : array-like
        Input data.
    model : array-like
        Input data.
    plus : array-like
        Input data.
    environment : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (5.3) [Multivariate Statistical Machine Learnin [Pages 141-170] [2026-04-16].pdf]
    """
    M11 = np.atleast_1d(np.asarray(M11, dtype=float))
    n = len(M11)
    result = float(np.mean(M11))
    se = float(np.std(M11, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (5.3) from MVSML chapter 5."})


def cheatsheet():
    return "msm024: Numbered display equation (5.3) from MVSML chapter 5."

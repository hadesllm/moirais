r"""Numbered display equation (15.4) from MVSML chapter 15.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_functional_regression_eq_15_4"]


def mvsml_functional_regression_eq_15_4(bY, b, The, ZAPC_RF, conventional, logistic):
    r"""
    Numbered display equation (15.4) from MVSML chapter 15.

    Formula: bY = (15.4) : b\theta + 0:5 b\mu, The ZAPC_RF as conventional logistic regression, the predicted values are probabilities and those probabilities are converted to a binary outcome if the probability is larger (or smaller) than some probability threshold (most of the time this threshold is 0.5). However, under the ZAPC_RF, instead of converting the probabilities to 0 and 1, we convert to zero if b\theta > 0:5

    Parameters
    ----------
    bY : array-like
        Input data.
    b : array-like
        Input data.
    The : array-like
        Input data.
    ZAPC_RF : array-like
        Input data.
    conventional : array-like
        Input data.
    logistic : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (15.4) [Multivariate Statistical Machine Learnin [Pages 633-681] [2026-04-16].pdf]
    r"""
    bY = np.atleast_1d(np.asarray(bY, dtype=float))
    n = len(bY)
    result = float(np.mean(bY))
    se = float(np.std(bY, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (15.4) from MVSML chapter 15."})


def cheatsheet():
    return "msm330: Numbered display equation (15.4) from MVSML chapter 15."

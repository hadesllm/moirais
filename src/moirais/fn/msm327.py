"""Numbered display equation (15.3) from MVSML chapter 15.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_functional_regression_eq_15_3"]


def mvsml_functional_regression_eq_15_3(LL, right, node, are, the, log):
    """
    Numbered display equation (15.3) from MVSML chapter 15.

    Formula: LL+ (right node) are the log-likelihood for each node. Once we have the estimates of \mu and \theta, the predicted values of Y under the ZAP_RF are obtained with   1  b\theta exp b\mu ( ) bY =

    Parameters
    ----------
    LL : array-like
        Input data.
    right : array-like
        Input data.
    node : array-like
        Input data.
    are : array-like
        Input data.
    the : array-like
        Input data.
    log : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (15.3) [Multivariate Statistical Machine Learnin [Pages 633-681] [2026-04-16].pdf]
    """
    LL = np.atleast_1d(np.asarray(LL, dtype=float))
    n = len(LL)
    result = float(np.mean(LL))
    se = float(np.std(LL, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (15.3) from MVSML chapter 15."})


def cheatsheet():
    return "msm327: Numbered display equation (15.3) from MVSML chapter 15."

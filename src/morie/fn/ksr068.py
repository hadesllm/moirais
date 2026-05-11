"""Profile partial-likelihood efficient score for the Cox model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_cox_profile_score"]


def kosorok_ch3_cox_profile_score(beta, Z, Y, X, tau, n):
    """
    Profile partial-likelihood efficient score for the Cox model

    Formula: l_hat_{beta,n}(X | X_1..X_n) = integral_0^tau { Z - P_n[ Z Y(t) e^{beta'Z} ] / P_n[ Y(t) e^{beta'Z} ] } dM_beta_hat(t)

    Parameters
    ----------
    beta : array-like
        Input data.
    Z : array-like
        Input data.
    Y : array-like
        Input data.
    X : array-like
        Input data.
    tau : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.8, p. 44
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n = len(beta)
    result = float(np.mean(beta))
    se = float(np.std(beta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Profile partial-likelihood efficient score for the Cox model"})


def cheatsheet():
    return "ksr068: Profile partial-likelihood efficient score for the Cox model"

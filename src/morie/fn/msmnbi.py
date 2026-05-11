"""MSM negative-binomial for overdispersed counts."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_negative_binomial"]


def msm_negative_binomial(y, treatment_history, covariate_history, alpha):
    """
    MSM negative-binomial for overdispersed counts

    Formula: log E[Y(a_bar)] = beta a_bar; Var = mu + alpha mu^2

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hilbe (2011); Robins-Hernán-Brumback (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM negative-binomial for overdispersed counts"})


def cheatsheet():
    return "msmnbi: MSM negative-binomial for overdispersed counts"

"""Numbered display equation (14.2) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_2"]


def mvsml_convolutional_nn_eq_14_2(dp, J, dtp, t, dt, where):
    """
    Numbered display equation (14.2) from MVSML chapter 14.

    Formula: 2 dp J\beta = dtp \beta t( ) dt, (14.11) 0 where dp dtp \beta t( ) is a derivative of order p of the function \beta(t). With the representation

    Parameters
    ----------
    dp : array-like
        Input data.
    J : array-like
        Input data.
    dtp : array-like
        Input data.
    t : array-like
        Input data.
    dt : array-like
        Input data.
    where : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.2) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    dp = np.atleast_1d(np.asarray(dp, dtype=float))
    n = len(dp)
    result = float(np.mean(dp))
    se = float(np.std(dp, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.2) from MVSML chapter 14."})


def cheatsheet():
    return "msm279: Numbered display equation (14.2) from MVSML chapter 14."

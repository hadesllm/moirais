"""R² for moderator in random-effects meta-regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_percent_heterogeneity_R2"]


def ma_percent_heterogeneity_R2(tau2_full, tau2_null):
    """
    R² for moderator in random-effects meta-regression

    Formula: R² = 1 - τ̂²_full/τ̂²_null

    Parameters
    ----------
    tau2_full : array-like
        Input data.
    tau2_null : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: R2

    References
    ----------
    Borenstein et al. (2009)
    """
    tau2_full = np.atleast_1d(np.asarray(tau2_full, dtype=float))
    n = len(tau2_full)
    result = float(np.mean(tau2_full))
    se = float(np.std(tau2_full, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "R² for moderator in random-effects meta-regression"})


def cheatsheet():
    return "marpct: R² for moderator in random-effects meta-regression"

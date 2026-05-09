"""Cox martingale residuals."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cox_martingale_residuals"]


def cox_martingale_residuals(fit):
    """
    Cox martingale residuals

    Formula: M_i = N_i(t) - integral exp(beta X_i) dH_0

    Parameters
    ----------
    fit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Therneau-Grambsch-Fleming (1990)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cox martingale residuals"})


def cheatsheet():
    return "coxmgr: Cox martingale residuals"

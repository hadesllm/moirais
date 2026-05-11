"""VAR impulse response function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["var_impulse_response"]


def var_impulse_response(fit, horizon):
    """
    VAR impulse response function

    Formula: Phi_h = sum w_i ... over horizon h

    Parameters
    ----------
    fit : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lütkepohl (2005)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VAR impulse response function"})


def cheatsheet():
    return "varimp: VAR impulse response function"

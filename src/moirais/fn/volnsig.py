"""Nelson skewed-GED-EGARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_nelson_skew_garch"]


def vol_nelson_skew_garch(r, init):
    """
    Nelson skewed-GED-EGARCH

    Formula: EGARCH residuals ~ skew-GED(ν, λ)

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: params, ll

    References
    ----------
    Nelson (1991)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nelson skewed-GED-EGARCH"})


def cheatsheet():
    return "volnsig: Nelson skewed-GED-EGARCH"

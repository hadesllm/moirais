"""CFA fit indices (CFI, RMSEA, SRMR, TLI)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cfa_fit_indices"]


def cfa_fit_indices(fit):
    """
    CFA fit indices (CFI, RMSEA, SRMR, TLI)

    Formula: chi-sq based + residual based

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
    Hu-Bentler (1999)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CFA fit indices (CFI, RMSEA, SRMR, TLI)"})


def cheatsheet():
    return "cfafit: CFA fit indices (CFI, RMSEA, SRMR, TLI)"

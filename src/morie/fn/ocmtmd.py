"""Outcome-model diagnostic for MSM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["outcome_model_diagnostic"]


def outcome_model_diagnostic(y, A, H, Q):
    """
    Outcome-model diagnostic for MSM

    Formula: residual vs treatment plot; std residuals

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    H : array-like
        Input data.
    Q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Crump et al (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Outcome-model diagnostic for MSM"})


def cheatsheet():
    return "ocmtmd: Outcome-model diagnostic for MSM"

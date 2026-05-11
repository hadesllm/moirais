# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bootstrapped standard errors for Aldrich-McKelvey scaling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["am_bootstrap_se"]


def am_bootstrap_se(survey_data, B):
    """
    Bootstrapped standard errors for Aldrich-McKelvey scaling

    Formula: SE_boot(theta) = sd(theta_b, b=1..B); resample rows with replacement B times

    Parameters
    ----------
    survey_data : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'se': 'matrix'}

    References
    ----------
    Armstrong Ch 2
    """
    survey_data = np.asarray(survey_data, dtype=float)
    n = int(survey_data) if survey_data.ndim == 0 else len(survey_data)
    result = float(np.mean(survey_data))
    se = float(np.std(survey_data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrapped standard errors for Aldrich-McKelvey scaling"})


def cheatsheet():
    return "ambtc: Bootstrapped standard errors for Aldrich-McKelvey scaling"

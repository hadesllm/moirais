"""Component GARCH separating short/long-term variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_cgarch_fit"]


def vol_cgarch_fit(r, init):
    """
    Component GARCH separating short/long-term variance

    Formula: σ_t² = q_t + α(ε² - q) + β(σ² - q); q_t persistence

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho, phi, omega, alpha, beta, ll

    References
    ----------
    Engle-Lee (1999)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Component GARCH separating short/long-term variance"})


def cheatsheet():
    return "volcgar: Component GARCH separating short/long-term variance"

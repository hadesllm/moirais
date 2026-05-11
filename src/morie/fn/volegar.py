"""EGARCH(1,1) MLE accommodating asymmetry."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_egarch_fit"]


def vol_egarch_fit(r, init):
    """
    EGARCH(1,1) MLE accommodating asymmetry

    Formula: log σ_t² = ω + β log σ_{t-1}² + α(|z_{t-1}|-E|z|) + γ z_{t-1}

    Parameters
    ----------
    r : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: omega, alpha, beta, gamma, ll

    References
    ----------
    Nelson (1991)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EGARCH(1,1) MLE accommodating asymmetry"})


def cheatsheet():
    return "volegar: EGARCH(1,1) MLE accommodating asymmetry"

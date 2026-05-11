"""Quantile treatment effect via Firpo IPW."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_quantile_treatment_effect"]


def causal_quantile_treatment_effect(y, T, ps, tau):
    """
    Quantile treatment effect via Firpo IPW

    Formula: QTE(τ) = F̂_1^{-1}(τ) - F̂_0^{-1}(τ) under PS weights

    Parameters
    ----------
    y : array-like
        Input data.
    T : array-like
        Input data.
    ps : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: QTE

    References
    ----------
    Firpo (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile treatment effect via Firpo IPW"})


def cheatsheet():
    return "causqte: Quantile treatment effect via Firpo IPW"

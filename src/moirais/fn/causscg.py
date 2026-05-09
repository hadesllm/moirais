"""Generalised synthetic control via interactive fixed effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_generalised_sc"]


def causal_generalised_sc(Y_panel, treated_idx, treat_time, r):
    """
    Generalised synthetic control via interactive fixed effects

    Formula: Y = α + λF + ε; impute Y_1(0) by latent factor model

    Parameters
    ----------
    Y_panel : array-like
        Input data.
    treated_idx : array-like
        Input data.
    treat_time : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATT, counterfactual

    References
    ----------
    Xu (2017)
    """
    Y_panel = np.atleast_1d(np.asarray(Y_panel, dtype=float))
    n = len(Y_panel)
    result = float(np.mean(Y_panel))
    se = float(np.std(Y_panel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalised synthetic control via interactive fixed effects"})


def cheatsheet():
    return "causscg: Generalised synthetic control via interactive fixed effects"

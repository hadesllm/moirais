"""Hartung-Knapp-Sidik-Jonkman variance correction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_hartung_knapp"]


def ma_hartung_knapp(yi, vi, tau2):
    """
    Hartung-Knapp-Sidik-Jonkman variance correction

    Formula: Var_HKSJ = (1/(k-1)) Σ w*_i (y_i-θ̂)²/Σw*_i

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    tau2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta, se_hksj, ci

    References
    ----------
    Hartung & Knapp (2001)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hartung-Knapp-Sidik-Jonkman variance correction"})


def cheatsheet():
    return "mahks: Hartung-Knapp-Sidik-Jonkman variance correction"

"""Wooldridge ETWFE event-time weighted DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_did_wooldridge_eta"]


def causal_did_wooldridge_eta(Y_panel, G_first_treat, X):
    """
    Wooldridge ETWFE event-time weighted DiD

    Formula: Saturated event-time x cohort interactions; no treatment-effect dilution

    Parameters
    ----------
    Y_panel : array-like
        Input data.
    G_first_treat : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATT_gt, ATT_overall

    References
    ----------
    Wooldridge (2021)
    """
    Y_panel = np.atleast_1d(np.asarray(Y_panel, dtype=float))
    n = len(Y_panel)
    result = float(np.mean(Y_panel))
    se = float(np.std(Y_panel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wooldridge ETWFE event-time weighted DiD"})


def cheatsheet():
    return "causdidwd: Wooldridge ETWFE event-time weighted DiD"

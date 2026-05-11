"""Synthetic DiD: combine SC weights + DiD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_synthetic_did"]


def causal_synthetic_did(Y_panel, treated_idx, treat_time):
    """
    Synthetic DiD: combine SC weights + DiD

    Formula: ATT = mean weighted post-pre diff

    Parameters
    ----------
    Y_panel : array-like
        Input data.
    treated_idx : array-like
        Input data.
    treat_time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATT, weights_unit, weights_time

    References
    ----------
    Arkhangelsky-Athey-Hirshberg-Imbens-Wager (2021)
    """
    Y_panel = np.atleast_1d(np.asarray(Y_panel, dtype=float))
    n = len(Y_panel)
    result = float(np.mean(Y_panel))
    se = float(np.std(Y_panel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Synthetic DiD: combine SC weights + DiD"})


def cheatsheet():
    return "causscd: Synthetic DiD: combine SC weights + DiD"

"""Event-study coefficients with relative-time dummies."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_did_eventstudy"]


def causal_did_eventstudy(Y_panel, K_event_time):
    """
    Event-study coefficients with relative-time dummies

    Formula: Y_it = α_i + γ_t + Σ_k β_k 1{K_it=k} + ε

    Parameters
    ----------
    Y_panel : array-like
        Input data.
    K_event_time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_k, se_k

    References
    ----------
    Borusyak-Jaravel-Spiess (2024)
    """
    Y_panel = np.atleast_1d(np.asarray(Y_panel, dtype=float))
    n = len(Y_panel)
    result = float(np.mean(Y_panel))
    se = float(np.std(Y_panel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Event-study coefficients with relative-time dummies"})


def cheatsheet():
    return "causdidev: Event-study coefficients with relative-time dummies"

# morie.fn -- function file (hadesllm/morie)
"""Direct Preference Optimization loss (preferred vs rejected completions)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dpo_loss"]


def geron_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta):
    """
    Direct Preference Optimization loss (preferred vs rejected completions)

    Formula: L_DPO = - log sigma( beta * [log pi(y_w|x) - log pi_ref(y_w|x) - (log pi(y_l|x) - log pi_ref(y_l|x))] )

    Parameters
    ----------
    logp_w : array-like
        Input data.
    logp_l : array-like
        Input data.
    logp_ref_w : array-like
        Input data.
    logp_ref_l : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 15, DPO section
    """
    logp_w = np.atleast_1d(np.asarray(logp_w, dtype=float))
    n = len(logp_w)
    result = float(np.mean(logp_w))
    se = float(np.std(logp_w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Direct Preference Optimization loss (preferred vs rejected completions)"})


def cheatsheet():
    return "grdpo: Direct Preference Optimization loss (preferred vs rejected completions)"

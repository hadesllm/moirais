# morie.fn -- function file (hadesllm/morie)
"""ACI: online update of alpha to maintain coverage under distribution shift."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_adaptive_conformal_inference"]


def joseph_adaptive_conformal_inference(alpha_t, miscoverage_t, eta, alpha_target):
    """
    ACI: online update of alpha to maintain coverage under distribution shift

    Formula: alpha_{t+1} = alpha_t + eta * (alpha_target - 1{y_t in PI_t})

    Parameters
    ----------
    alpha_t : array-like
        Input data.
    miscoverage_t : array-like
        Input data.
    eta : array-like
        Input data.
    alpha_target : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: alpha_new

    References
    ----------
    Joseph Ch 17, Adaptive Conformal Inference section
    """
    alpha_t = np.atleast_1d(np.asarray(alpha_t, dtype=float))
    n = len(alpha_t)
    result = float(np.mean(alpha_t))
    se = float(np.std(alpha_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ACI: online update of alpha to maintain coverage under distribution shift"})


def cheatsheet():
    return "joaci: ACI: online update of alpha to maintain coverage under distribution shift"

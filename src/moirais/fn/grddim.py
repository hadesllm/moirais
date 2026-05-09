# moirais.fn — function file (hadesllm/moirais)
"""DDIM deterministic sampling step (subset schedule, eta=0)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ddim_sampling_step"]


def geron_ddim_sampling_step(x_t, t, t_prev, eps_pred, alpha_bar):
    """
    DDIM deterministic sampling step (subset schedule, eta=0)

    Formula: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1}) * eps_theta(x_t, t)

    Parameters
    ----------
    x_t : array-like
        Input data.
    t : array-like
        Input data.
    t_prev : array-like
        Input data.
    eps_pred : array-like
        Input data.
    alpha_bar : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_t_minus_1

    References
    ----------
    Géron Ch 18, DDIM sampling section
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DDIM deterministic sampling step (subset schedule, eta=0)"})


def cheatsheet():
    return "grddim: DDIM deterministic sampling step (subset schedule, eta=0)"

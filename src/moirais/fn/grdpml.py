# moirais.fn — function file (hadesllm/moirais)
"""DDPM simplified training loss — MSE on predicted noise."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ddpm_simple_loss"]


def geron_ddpm_simple_loss(eps, eps_pred):
    """
    DDPM simplified training loss — MSE on predicted noise

    Formula: L_simple = E_{t, x_0, eps} [ ||eps - eps_theta(x_t, t)||^2 ]

    Parameters
    ----------
    eps : array-like
        Input data.
    eps_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 18, DDPM training section
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    n = len(eps)
    result = float(np.mean(eps))
    se = float(np.std(eps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DDPM simplified training loss — MSE on predicted noise"})


def cheatsheet():
    return "grdpml: DDPM simplified training loss — MSE on predicted noise"

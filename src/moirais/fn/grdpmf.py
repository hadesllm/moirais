# moirais.fn — function file (hadesllm/moirais)
"""DDPM forward (noising) process q(x_t | x_0)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ddpm_forward_process"]


def geron_ddpm_forward_process(x0, t, alpha_bar):
    """
    DDPM forward (noising) process q(x_t | x_0)

    Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps ~ N(0, I)

    Parameters
    ----------
    x0 : array-like
        Input data.
    t : array-like
        Input data.
    alpha_bar : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_t, noise

    References
    ----------
    Géron Ch 18, DDPM forward diffusion section
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = len(x0)
    result = float(np.mean(x0))
    se = float(np.std(x0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DDPM forward (noising) process q(x_t | x_0)"})


def cheatsheet():
    return "grdpmf: DDPM forward (noising) process q(x_t | x_0)"

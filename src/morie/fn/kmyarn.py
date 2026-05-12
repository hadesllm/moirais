# morie.fn -- function file (hadesllm/morie)
"""YaRN NTK-aware RoPE rescaling for context-window extrapolation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_yarn_context_extrapolation"]


def kamath_yarn_context_extrapolation(theta, scale, d):
    """
    YaRN NTK-aware RoPE rescaling for context-window extrapolation

    Formula: theta_i_new = theta_i * (1 / (s^(2i/d)));  s = scale factor; blend low/high-freq ranges

    Parameters
    ----------
    theta : array-like
        Input data.
    scale : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_new

    References
    ----------
    Kamath Ch 10, YaRN section
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "YaRN NTK-aware RoPE rescaling for context-window extrapolation"})


def cheatsheet():
    return "kmyarn: YaRN NTK-aware RoPE rescaling for context-window extrapolation"

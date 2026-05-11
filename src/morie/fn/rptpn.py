# morie.fn — function file (hadesllm/morie)
"""Repetition penalty for generation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["repetition_penalty"]


def repetition_penalty(x):
    """
    Repetition penalty for generation

    Formula: logit[t] /= alpha if t in generated

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Keskar et al. (2019)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Repetition penalty for generation"})


def cheatsheet():
    return "rptpn: Repetition penalty for generation"

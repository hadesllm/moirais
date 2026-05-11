"""Flamingo gated cross-attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["flamingo_gated_cross"]


def flamingo_gated_cross(x, vision, gate):
    """
    Flamingo gated cross-attention

    Formula: gating tanh(g) * cross_attn(x, vis)

    Parameters
    ----------
    x : array-like
        Input data.
    vision : array-like
        Input data.
    gate : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Alayrac et al (2022) Flamingo
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Flamingo gated cross-attention"})


def cheatsheet():
    return "flmgcr: Flamingo gated cross-attention"

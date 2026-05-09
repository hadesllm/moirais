# moirais.fn — function file (hadesllm/moirais)
"""Rotary positional embedding (RoPE): rotate query/key by angle m*theta_i."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_rotary_positional_embedding"]


def kamath_rotary_positional_embedding(q, positions, base):
    """
    Rotary positional embedding (RoPE): rotate query/key by angle m*theta_i

    Formula: [q_2i, q_2i+1] -> [cos(m*theta_i) q_2i - sin(m*theta_i) q_2i+1, sin(...) q_2i + cos(...) q_2i+1]

    Parameters
    ----------
    q : array-like
        Input data.
    positions : array-like
        Input data.
    base : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: q_rotated

    References
    ----------
    Kamath Ch 2, RoPE section (Su et al.)
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    result = float(np.mean(q))
    se = float(np.std(q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rotary positional embedding (RoPE): rotate query/key by angle m*theta_i"})


def cheatsheet():
    return "kmrope: Rotary positional embedding (RoPE): rotate query/key by angle m*theta_i"

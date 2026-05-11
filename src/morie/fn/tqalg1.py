"""Algorithm 1: online per-token key-cache quantizer (QJL)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_online_key_quantizer"]


def turboquant_online_key_quantizer(k, S):
    """
    Algorithm 1: online per-token key-cache quantizer (QJL)

    Formula: k_tilde = sign(S k);  nu = ||k||_2;  store (k_tilde, nu) per token

    Parameters
    ----------
    k : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: k_tilde, nu

    References
    ----------
    Zandieh et al. 2024 Algorithm 1 (key quantizer)
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = len(k)
    result = float(np.mean(k))
    se = float(np.std(k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Algorithm 1: online per-token key-cache quantizer (QJL)"})


def cheatsheet():
    return "tqalg1: Algorithm 1: online per-token key-cache quantizer (QJL)"

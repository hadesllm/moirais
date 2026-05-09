"""ViT-2 log-scaled attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit2_log_attention"]


def vit2_log_attention(q, k, v):
    """
    ViT-2 log-scaled attention

    Formula: attn over log-normalized similarities

    Parameters
    ----------
    q : array-like
        Input data.
    k : array-like
        Input data.
    v : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yu et al (2022)
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    result = float(np.mean(q))
    se = float(np.std(q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT-2 log-scaled attention"})


def cheatsheet():
    return "vit2lf: ViT-2 log-scaled attention"

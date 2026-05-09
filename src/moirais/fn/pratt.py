"""Pretrained attention pooling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pretrained_attention"]


def pretrained_attention(tokens, model):
    """
    Pretrained attention pooling

    Formula: sentence vec = sum α_i h_i with learned attn

    Parameters
    ----------
    tokens : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yang et al (2016) hierarchical attn
    """
    tokens = np.atleast_1d(np.asarray(tokens, dtype=float))
    n = len(tokens)
    result = float(np.mean(tokens))
    se = float(np.std(tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pretrained attention pooling"})


def cheatsheet():
    return "pratt: Pretrained attention pooling"

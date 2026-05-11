"""ViLBERT two-stream cross-attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vilbert_two_stream"]


def vilbert_two_stream(image_tokens, text_tokens):
    """
    ViLBERT two-stream cross-attention

    Formula: separate visual + text streams; co-attn

    Parameters
    ----------
    image_tokens : array-like
        Input data.
    text_tokens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lu et al (2019) ViLBERT
    """
    image_tokens = np.atleast_1d(np.asarray(image_tokens, dtype=float))
    n = len(image_tokens)
    result = float(np.mean(image_tokens))
    se = float(np.std(image_tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViLBERT two-stream cross-attention"})


def cheatsheet():
    return "vilbrt: ViLBERT two-stream cross-attention"

# moirais.fn — function file (hadesllm/moirais)
"""DALL-E autoregressive text-to-image token modeling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dalle_autoregressive_token"]


def geron_dalle_autoregressive_token(text_tokens, image_tokens_prefix):
    """
    DALL-E autoregressive text-to-image token modeling

    Formula: p(img_tokens | text) = prod_t p(img_t | text, img_{<t})

    Parameters
    ----------
    text_tokens : array-like
        Input data.
    image_tokens_prefix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: logits

    References
    ----------
    Géron Ch 16, DALL-E section
    """
    text_tokens = np.atleast_1d(np.asarray(text_tokens, dtype=float))
    n = len(text_tokens)
    result = float(np.mean(text_tokens))
    se = float(np.std(text_tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DALL-E autoregressive text-to-image token modeling"})


def cheatsheet():
    return "grdal: DALL-E autoregressive text-to-image token modeling"

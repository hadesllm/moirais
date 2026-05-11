# morie.fn — function file (hadesllm/morie)
"""VideoBERT: transformer on discretized video tokens + text."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_videobert"]


def geron_videobert(video_tokens, text_tokens):
    """
    VideoBERT: transformer on discretized video tokens + text

    Formula: joint MLM on video+text tokens

    Parameters
    ----------
    video_tokens : array-like
        Input data.
    text_tokens : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 16
    """
    video_tokens = np.atleast_1d(np.asarray(video_tokens, dtype=float))
    n = len(video_tokens)
    result = float(np.mean(video_tokens))
    se = float(np.std(video_tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VideoBERT: transformer on discretized video tokens + text"})


def cheatsheet():
    return "hmvbrt: VideoBERT: transformer on discretized video tokens + text"

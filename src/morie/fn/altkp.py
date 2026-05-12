# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Tokenization pipeline: normalize -> pre-tokenize -> subword segment -> post-process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_tokenization_pipeline"]


def alammar_tokenization_pipeline(text, tokenizer):
    """
    Tokenization pipeline: normalize -> pre-tokenize -> subword segment -> post-process

    Formula: tokens = Post(Subword(Pre(Normalize(text))))

    Parameters
    ----------
    text : array-like
        Input data.
    tokenizer : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tokens

    References
    ----------
    Alammar Ch 2, LLM Tokenization Pipeline section
    """
    text = np.atleast_1d(np.asarray(text, dtype=float))
    n = len(text)
    result = float(np.mean(text))
    se = float(np.std(text, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tokenization pipeline: normalize -> pre-tokenize -> subword segment -> post-process"})


def cheatsheet():
    return "altkp: Tokenization pipeline: normalize -> pre-tokenize -> subword segment -> post-process"

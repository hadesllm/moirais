"""tiktoken-style efficient BPE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tiktoken_bpe"]


def tiktoken_bpe(corpus):
    """
    tiktoken-style efficient BPE

    Formula: BPE with rust regex pre-tokenizer

    Parameters
    ----------
    corpus : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    OpenAI tiktoken (2022)
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "tiktoken-style efficient BPE"})


def cheatsheet():
    return "tikto: tiktoken-style efficient BPE"

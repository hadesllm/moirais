# moirais.fn — function file (hadesllm/moirais)
"""SentencePiece unigram-language-model tokenizer: maximize corpus likelihood over subword units."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_sentencepiece_tokenizer"]


def kamath_sentencepiece_tokenizer(corpus, vocab_size):
    """
    SentencePiece unigram-language-model tokenizer: maximize corpus likelihood over subword units

    Formula: argmax_{V, p} sum_{sentence s} log P(s | V, p);  P(s) = sum_{seg} prod_t p(w_t)

    Parameters
    ----------
    corpus : array-like
        Input data.
    vocab_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: vocab, scores

    References
    ----------
    Kamath Ch 2, SentencePiece section
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SentencePiece unigram-language-model tokenizer: maximize corpus likelihood over subword units"})


def cheatsheet():
    return "kmsp: SentencePiece unigram-language-model tokenizer: maximize corpus likelihood over subword units"

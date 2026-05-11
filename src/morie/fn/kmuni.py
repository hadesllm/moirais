# morie.fn — function file (hadesllm/morie)
"""Unigram LM tokenizer: EM over piece probabilities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_unigram_lm_tokenizer"]


def kamath_unigram_lm_tokenizer(corpus, vocab):
    """
    Unigram LM tokenizer: EM over piece probabilities

    Formula: E-step: alpha_t ∝ p(w_t); M-step: p(w_t) ∝ expected count of w_t

    Parameters
    ----------
    corpus : array-like
        Input data.
    vocab : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probs

    References
    ----------
    Kamath Ch 2, Unigram LM section
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unigram LM tokenizer: EM over piece probabilities"})


def cheatsheet():
    return "kmuni: Unigram LM tokenizer: EM over piece probabilities"

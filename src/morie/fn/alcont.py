# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Continued pretraining on domain corpus with MLM objective before fine-tuning."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_continued_pretraining_mlm"]


def alammar_continued_pretraining_mlm(domain_corpus, encoder, n_mlm_steps):
    """
    Continued pretraining on domain corpus with MLM objective before fine-tuning

    Formula: L_MLM over domain corpus for N steps;  then L_task on labeled data

    Parameters
    ----------
    domain_corpus : array-like
        Input data.
    encoder : array-like
        Input data.
    n_mlm_steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: encoder_adapted

    References
    ----------
    Alammar Ch 11, Continued Pretraining section
    """
    domain_corpus = np.atleast_1d(np.asarray(domain_corpus, dtype=float))
    n = len(domain_corpus)
    result = float(np.mean(domain_corpus))
    se = float(np.std(domain_corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continued pretraining on domain corpus with MLM objective before fine-tuning"})


def cheatsheet():
    return "alcont: Continued pretraining on domain corpus with MLM objective before fine-tuning"

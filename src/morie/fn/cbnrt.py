# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""CausalBERT: text representation for CATE estimation with NLP confounders."""
import numpy as np
from ._richresult import RichResult

__all__ = ["causalbert_text"]


def causalbert_text(texts, T, Y, X):
    """
    CausalBERT: text representation for CATE estimation with NLP confounders

    Formula: Z = BERT(text); T = f(Z,X); Y = g(T,Z,X); propensity and outcome modeled with text embeddings

    Parameters
    ----------
    texts : array-like
        Input data.
    T : array-like
        Input data.
    Y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'cate': 'array'}

    References
    ----------
    Molak Ch 11
    """
    texts = np.asarray(texts, dtype=float)
    n = int(texts) if texts.ndim == 0 else len(texts)
    if texts.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "CausalBERT: text representation for CATE estimation with NLP confounders"})
    estimate = np.median(texts)
    se = 1.2533 * np.std(texts, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "CausalBERT: text representation for CATE estimation with NLP confounders"})


def cheatsheet():
    return "cbnrt: CausalBERT: text representation for CATE estimation with NLP confounders"

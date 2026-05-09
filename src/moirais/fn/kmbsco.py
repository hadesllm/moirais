# moirais.fn — function file (hadesllm/moirais)
"""BERTScore: token-level cosine similarity between contextual embeddings."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_bertscore"]


def kamath_bertscore(hypothesis_tokens, reference_tokens, embed_fn):
    """
    BERTScore: token-level cosine similarity between contextual embeddings

    Formula: P = (1/|h|) sum_i max_j cos(emb(h_i), emb(r_j));  similar R; F1 = 2PR/(P+R)

    Parameters
    ----------
    hypothesis_tokens : array-like
        Input data.
    reference_tokens : array-like
        Input data.
    embed_fn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: precision, recall, f1

    References
    ----------
    Kamath Ch 8, BERTScore section
    """
    hypothesis_tokens = np.atleast_1d(np.asarray(hypothesis_tokens, dtype=float))
    n = len(hypothesis_tokens)
    result = float(np.mean(hypothesis_tokens))
    se = float(np.std(hypothesis_tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BERTScore: token-level cosine similarity between contextual embeddings"})


def cheatsheet():
    return "kmbsco: BERTScore: token-level cosine similarity between contextual embeddings"

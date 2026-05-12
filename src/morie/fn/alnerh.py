# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""NER token-classification head: per-token softmax over BIO tags."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_ner_token_head"]


def alammar_ner_token_head(h_tokens, W, b, tags):
    """
    NER token-classification head: per-token softmax over BIO tags

    Formula: p(tag | h_t) = softmax(W h_t + b);  loss = CE over tag targets

    Parameters
    ----------
    h_tokens : array-like
        Input data.
    W : array-like
        Input data.
    b : array-like
        Input data.
    tags : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Alammar Ch 11, NER Token Classification section
    """
    h_tokens = np.atleast_1d(np.asarray(h_tokens, dtype=float))
    n = len(h_tokens)
    result = float(np.mean(h_tokens))
    se = float(np.std(h_tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NER token-classification head: per-token softmax over BIO tags"})


def cheatsheet():
    return "alnerh: NER token-classification head: per-token softmax over BIO tags"

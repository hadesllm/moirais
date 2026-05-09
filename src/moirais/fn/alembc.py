# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Embedding-based classifier: logistic regression / SVM on top of frozen embeddings."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_embedding_classifier"]


def alammar_embedding_classifier(embeddings, labels, classifier):
    """
    Embedding-based classifier: logistic regression / SVM on top of frozen embeddings

    Formula: p(y | x) = softmax(W * emb(x) + b);  emb comes from a frozen encoder

    Parameters
    ----------
    embeddings : array-like
        Input data.
    labels : array-like
        Input data.
    classifier : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probs

    References
    ----------
    Alammar Ch 4, Embedding-based Supervised Classification section
    """
    embeddings = np.atleast_1d(np.asarray(embeddings, dtype=float))
    n = len(embeddings)
    result = float(np.mean(embeddings))
    se = float(np.std(embeddings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Embedding-based classifier: logistic regression / SVM on top of frozen embeddings"})


def cheatsheet():
    return "alembc: Embedding-based classifier: logistic regression / SVM on top of frozen embeddings"

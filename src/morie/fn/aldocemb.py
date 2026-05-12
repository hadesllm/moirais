# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Document-level embedding via mean-pool over contextual token vectors."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_document_embedding_pool"]


def alammar_document_embedding_pool(token_embeddings, attention_mask):
    """
    Document-level embedding via mean-pool over contextual token vectors

    Formula: d_vec = (1/L) sum_{i=1..L} h_i

    Parameters
    ----------
    token_embeddings : array-like
        Input data.
    attention_mask : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: d_vec

    References
    ----------
    Alammar Ch 2, Document-Level Embeddings section
    """
    token_embeddings = np.atleast_1d(np.asarray(token_embeddings, dtype=float))
    n = len(token_embeddings)
    result = float(np.mean(token_embeddings))
    se = float(np.std(token_embeddings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Document-level embedding via mean-pool over contextual token vectors"})


def cheatsheet():
    return "aldocemb: Document-level embedding via mean-pool over contextual token vectors"

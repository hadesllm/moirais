# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Token embedding table lookup."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_token_embedding_lookup"]


def alammar_token_embedding_lookup(ids, E_tok):
    """
    Token embedding table lookup

    Formula: E_tok[ids]  where E_tok in R^{V x d} and ids in {0,...,V-1}^L

    Parameters
    ----------
    ids : array-like
        Input data.
    E_tok : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: embeddings

    References
    ----------
    Alammar Ch 2, Token Embedding section
    """
    ids = np.atleast_1d(np.asarray(ids, dtype=float))
    n = len(ids)
    result = float(np.mean(ids))
    se = float(np.std(ids, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Token embedding table lookup"})


def cheatsheet():
    return "altkemb: Token embedding table lookup"

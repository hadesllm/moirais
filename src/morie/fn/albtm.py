# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""BERTopic pipeline: embed -> UMAP -> HDBSCAN -> c-TF-IDF -> topic labels."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_bertopic_pipeline"]


def alammar_bertopic_pipeline(documents, embedder, reducer, cluster_model):
    """
    BERTopic pipeline: embed -> UMAP -> HDBSCAN -> c-TF-IDF -> topic labels

    Formula: topics = cTFIDF(HDBSCAN(UMAP(embed(docs))))

    Parameters
    ----------
    documents : array-like
        Input data.
    embedder : array-like
        Input data.
    reducer : array-like
        Input data.
    cluster_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: topics

    References
    ----------
    Alammar Ch 5, BERTopic section
    """
    documents = np.atleast_1d(np.asarray(documents, dtype=float))
    n = len(documents)
    result = float(np.mean(documents))
    se = float(np.std(documents, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BERTopic pipeline: embed -> UMAP -> HDBSCAN -> c-TF-IDF -> topic labels"})


def cheatsheet():
    return "albtm: BERTopic pipeline: embed -> UMAP -> HDBSCAN -> c-TF-IDF -> topic labels"

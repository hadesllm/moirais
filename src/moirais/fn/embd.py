# moirais.fn — function file (hadesllm/moirais)
"""Token/word embedding similarity via cosine distance."""

import numpy as np

from ._containers import DescriptiveResult


def embedding_similarity(embeddings, labels=None, metric="cosine"):
    """
    Compute pairwise similarity between embedding vectors.

    :param embeddings: (n, d) embedding matrix.
    :param labels: (n,) optional labels for rows.
    :param metric: 'cosine' or 'euclidean'.
    :return: DescriptiveResult with similarity matrix, nearest neighbors.

    References
    ----------
    Mikolov T et al. (2013). Efficient Estimation of Word Representations
    in Vector Space. ICLR Workshop.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    if E.ndim == 1:
        E = E[None, :]
    n, d = E.shape

    if metric == "cosine":
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        E_norm = E / norms
        sim = E_norm @ E_norm.T
    else:
        sq = np.sum(E**2, axis=1)
        dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * E @ E.T, 0))
        sim = 1 / (1 + dist)

    np.fill_diagonal(sim, 0)
    nearest = np.argmax(sim, axis=1)
    if labels is not None:
        labels = list(labels)
        nn_labels = [labels[i] for i in nearest]
    else:
        nn_labels = nearest.tolist()

    return DescriptiveResult(
        name="embedding_similarity",
        value=float(np.mean(sim[sim > 0])) if np.any(sim > 0) else 0.0,
        extra={
            "similarity_matrix": sim,
            "nearest_neighbors": nn_labels,
            "metric": metric,
            "n": n,
            "d": d,
        },
    )


def cheatsheet() -> str:
    return "embedding_similarity({}) -> Token/word embedding similarity via cosine distance."

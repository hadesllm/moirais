"""t-SNE for non-linear dimension reduction / visualisation."""
import numpy as np

from ._richresult import RichResult

__all__ = ["tsne_reduction"]


def tsne_reduction(x, *, n_components=2, perplexity=30.0,
                    learning_rate="auto", n_iter=1000, seed=0):
    """t-SNE embedding via sklearn.manifold.TSNE.

    Minimises KL(P||Q) where p_ij are Gaussian joint probabilities in
    input space and q_ij = (1+||y_i - y_j||^2)^-1 / Z is Student-t in
    embedding space.

    Parameters
    ----------
    x : array-like (n, p).
    n_components : int
        Embedding dimension (2 or 3 typical).
    perplexity : float
        Effective number of neighbours.
    learning_rate : "auto" | float.
    n_iter : int
        Max optimisation iterations.
    seed : int
        random_state.

    Returns
    -------
    RichResult with payload: estimate (embedding shape), embedding
    (n x n_components), kl_divergence, perplexity, n, method.
    """
    from sklearn.manifold import TSNE

    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    # sklearn renamed n_iter -> max_iter at v1.5
    try:
        ts = TSNE(n_components=n_components, perplexity=perplexity,
                  learning_rate=learning_rate, max_iter=n_iter,
                  random_state=seed, init="pca")
    except TypeError:
        ts = TSNE(n_components=n_components, perplexity=perplexity,
                  learning_rate=learning_rate, n_iter=n_iter,
                  random_state=seed, init="pca")
    emb = ts.fit_transform(X)
    return RichResult(payload={
        "estimate": list(emb.shape),
        "embedding": emb.tolist(),
        "kl_divergence": float(ts.kl_divergence_),
        "perplexity": float(perplexity),
        "n_components": int(n_components),
        "n": int(n),
        "method": "t-SNE (van der Maaten 2008)",
    })


def cheatsheet():
    return "tsnrd: t-SNE non-linear embedding"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Two well-separated Gaussian blobs
    X = np.vstack([rng.normal(loc=-3, size=(50, 5)),
                   rng.normal(loc=+3, size=(50, 5))])
    r = tsne_reduction(X, n_components=2, perplexity=10.0, n_iter=500, seed=0)
    print("embedding shape:", r.estimate)
    print("KL divergence:", r.kl_divergence)

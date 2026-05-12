"""PCA via SVD for dimension reduction."""
import numpy as np

from ._richresult import RichResult

__all__ = ["pca_dimension_reduction"]


def pca_dimension_reduction(x, *, n_components=None, seed=0):
    """Principal Component Analysis via sklearn.decomposition.PCA.

    Decomposes the centred design matrix X = U Sigma V', keeps the top k
    components, and returns the explained-variance breakdown.

    Parameters
    ----------
    x : array-like (n, p).
    n_components : int or None
        Number of components.  None -> min(n, p).
    seed : int
        random_state (svd_solver="auto" honours it for randomized SVD).

    Returns
    -------
    RichResult with payload: estimate (explained variance ratio of first
    component), components (V' rows), explained_variance,
    explained_variance_ratio, singular_values, scores (X @ V), n, method.
    """
    from sklearn.decomposition import PCA

    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    k = n_components if n_components is not None else min(n, p)
    pca = PCA(n_components=k, random_state=seed, svd_solver="full")
    scores = pca.fit_transform(X)
    return RichResult(payload={
        "estimate": float(pca.explained_variance_ratio_[0]),
        "components": pca.components_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
        "scores": scores.tolist(),
        "n_components": int(k),
        "n": int(n),
        "method": "PCA via SVD",
    })


def cheatsheet():
    return "pcadm: PCA dimension reduction (SVD)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Build a dataset where the first PC dominates
    z = rng.normal(size=(200, 1))
    X = z @ np.array([[1.0, 2.0, -1.0]]) + 0.1 * rng.normal(size=(200, 3))
    r = pca_dimension_reduction(X, n_components=3)
    print("explained variance ratio:", r.explained_variance_ratio)
    print("singular values:", r.singular_values)
    print("first-PC ratio:", r.estimate)

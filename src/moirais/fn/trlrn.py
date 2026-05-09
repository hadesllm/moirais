"""Transfer learning via subspace alignment."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what they grow beyond."


def transfer_learn(X_source, y_source, X_target, n_components=10, **kwargs) -> DescriptiveResult:
    """Simple transfer learning via subspace alignment (Fernando et al., 2013).

    Projects source and target into PCA subspaces, aligns them via
    the alignment matrix M = Ps^T Pt, then trains a nearest-centroid
    classifier in the aligned space.

    Parameters
    ----------
    X_source : array-like of shape (n_s, p)
    y_source : array-like of shape (n_s,)
    X_target : array-like of shape (n_t, p)
    n_components : int
        Subspace dimensionality (default 10).

    Returns
    -------
    DescriptiveResult

    References
    ----------
    Fernando, B., Habrard, A., Sebban, M. & Tuytelaars, T. (2013).
        Unsupervised visual domain adaptation using subspace alignment.
        *ICCV*, 2960--2967.
    """
    X_s = np.asarray(X_source, dtype=float)
    y_s = np.asarray(y_source).ravel()
    X_t = np.asarray(X_target, dtype=float)

    d = min(n_components, X_s.shape[1], X_t.shape[1])

    def _pca_basis(X, k):
        mu = X.mean(axis=0)
        Xc = X - mu
        cov = Xc.T @ Xc / max(len(X) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(-eigvals)[:k]
        return eigvecs[:, idx], mu

    Ps, mu_s = _pca_basis(X_s, d)
    Pt, mu_t = _pca_basis(X_t, d)

    M = Ps.T @ Pt

    X_s_aligned = (X_s - mu_s) @ Ps @ M
    X_t_proj = (X_t - mu_t) @ Pt

    classes = np.unique(y_s)
    centroids = np.array([X_s_aligned[y_s == c].mean(axis=0) for c in classes])
    dists = np.array([np.sum((X_t_proj - c) ** 2, axis=1) for c in centroids])
    predictions = classes[np.argmin(dists, axis=0)]

    alignment_quality = float(np.mean(np.abs(np.diag(M))))

    return DescriptiveResult(
        name="transfer_learn",
        value=alignment_quality,
        extra={
            "predictions": predictions,
            "alignment_matrix": M,
            "alignment_quality": alignment_quality,
            "n_components": d,
            "source_basis": Ps,
            "target_basis": Pt,
        },
    )


trlrn = transfer_learn


def cheatsheet() -> str:
    return "transfer_learn({}) -> Transfer learning via subspace alignment."

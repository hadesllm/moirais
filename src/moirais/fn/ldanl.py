# moirais.fn — function file (hadesllm/moirais)
"""Linear discriminant analysis."""

from __future__ import annotations

import numpy as np

from ._containers import LdaRes


def lda(
    data: np.ndarray,
    labels: np.ndarray,
    n_components: int | None = None,
) -> LdaRes:
    """Fisher's Linear Discriminant Analysis.

    Parameters
    ----------
    data : ndarray (n, p)
        Feature matrix.
    labels : ndarray (n,)
        Class labels (integer or string).
    n_components : int, optional
        Number of discriminant components. Defaults to min(p, n_classes - 1).

    Returns
    -------
    LdaRes
    """
    X = np.asarray(data, dtype=np.float64)
    y = np.asarray(labels)
    n, p = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    overall_mean = X.mean(axis=0)
    Sw = np.zeros((p, p))
    Sb = np.zeros((p, p))

    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        mc = Xc.mean(axis=0)
        diff = Xc - mc
        Sw += diff.T @ diff
        d = (mc - overall_mean).reshape(-1, 1)
        Sb += nc * (d @ d.T)

    Sw += np.eye(p) * 1e-8

    eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(Sw) @ Sb)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k = n_components or min(p, n_classes - 1)
    eigvals_k = eigvals[:k]
    components = eigvecs[:, :k]
    total = eigvals.sum()
    ratio = eigvals_k / total if total > 0 else np.zeros(k)
    projected = X @ components

    return LdaRes(
        components=components,
        explained_variance_ratio=ratio,
        projected=projected,
    )


ldanl = lda


def cheatsheet() -> str:
    return "lda({}) -> Fisher's linear discriminant analysis."

# morie.fn — function file (hadesllm/morie)
"""Linear Discriminant Analysis (Fisher's LDA)."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import LdaRes


def lda_(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int | None = None,
) -> LdaRes:
    r"""Fisher's Linear Discriminant Analysis.

    Projects data onto the directions that maximise the ratio of
    between-class to within-class scatter:

    .. math::

        \mathbf{w}^* = \arg\max_{\mathbf{w}}
        \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}

    Parameters
    ----------
    X : ndarray (n, p)
        Feature matrix.
    y : ndarray (n,)
        Class labels (integer or string).
    n_components : int, optional
        Number of discriminant components.  *None* keeps min(n_classes - 1, p).

    Returns
    -------
    LdaRes
        ``components`` (p x k), ``explained_variance_ratio``,
        ``projected`` (n x k).

    References
    ----------
    Fisher, R. A. (1936). The use of multiple measurements in taxonomic
    problems. *Annals of Eugenics*, 7(2), 179-188.
    DOI: 10.1111/j.1469-1809.1936.tb02137.x
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, p = X.shape

    classes = np.unique(y)
    n_classes = len(classes)
    k = n_components if n_components is not None else min(n_classes - 1, p)
    k = min(k, n_classes - 1, p)
    k = max(k, 1)

    overall_mean = X.mean(axis=0)

    S_W = np.zeros((p, p))
    S_B = np.zeros((p, p))

    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]
        mean_c = X_c.mean(axis=0)

        # Within-class scatter
        diff = X_c - mean_c
        S_W += diff.T @ diff

        # Between-class scatter
        mean_diff = (mean_c - overall_mean).reshape(-1, 1)
        S_B += n_c * (mean_diff @ mean_diff.T)

    # Regularise S_W for numerical stability
    S_W += np.eye(p) * 1e-10

    # Solve generalised eigenvalue problem S_B w = lambda S_W w
    # Equivalent to S_W^{-1} S_B w = lambda w
    try:
        S_W_inv = np.linalg.inv(S_W)
    except np.linalg.LinAlgError:
        S_W_inv = np.linalg.pinv(S_W)

    M = S_W_inv @ S_B
    eigvals, eigvecs = np.linalg.eigh(M)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]

    components = eigvecs[:, :k]
    total = np.sum(eigvals)
    ratio = eigvals[:k] / total if total > 0 else np.zeros(k)

    projected = X @ components

    return LdaRes(
        components=components,
        explained_variance_ratio=ratio,
        projected=projected,
    )


def cheatsheet() -> str:
    return "lda_({}) -> Linear Discriminant Analysis (Fisher's LDA)."

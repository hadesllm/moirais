# morie.fn -- function file (hadesllm/morie)
"""Transfer features from a source domain to a target domain via."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def absorption_features(
    source_embeddings: np.ndarray,
    target_data: np.ndarray,
    *,
    n_components: int | None = None,
    regularize: float = 1e-6,
) -> DescriptiveResult:
    """Transfer features from a source domain to a target domain via
    projection onto the principal subspace of source embeddings.

    Fits PCA on source embeddings and projects target data, implementing
    a simple feature extraction form of transfer learning.

    Parameters
    ----------
    source_embeddings : np.ndarray
        (n_source x p) feature matrix from the source domain.
    target_data : np.ndarray
        (n_target x p) raw features from the target domain.
    n_components : int or None
        Dimensions to retain.  Default: min(n_source, p).
    regularize : float
        Ridge regularization for covariance estimation.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``projected`` (n_target x n_components),
        ``explained_variance_ratio``, ``components`` (loadings).
    """
    S = np.asarray(source_embeddings, dtype=float)
    T = np.asarray(target_data, dtype=float)
    if S.ndim != 2 or T.ndim != 2:
        raise ValueError("Both inputs must be 2D")
    if S.shape[1] != T.shape[1]:
        raise ValueError("source and target must have same number of features")

    p = S.shape[1]
    if n_components is None:
        n_components = min(S.shape[0], p)
    n_components = min(n_components, p)

    mu = S.mean(axis=0)
    S_centered = S - mu
    cov = (S_centered.T @ S_centered) / max(S.shape[0] - 1, 1) + regularize * np.eye(p)

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(-eigvals)[:n_components]
    eigvals = eigvals[idx]
    components = eigvecs[:, idx]

    total_var = eigvals.sum()
    evr = eigvals / total_var if total_var > 0 else eigvals

    T_centered = T - mu
    projected = T_centered @ components

    return DescriptiveResult(
        name="absorption_features",
        value={
            "projected": projected,
            "explained_variance_ratio": evr,
            "components": components,
        },
        extra={"n_components": n_components, "n_source": S.shape[0], "n_target": T.shape[0]},
    )


absfea = absorption_features


def cheatsheet() -> str:
    return 'absorption_features({}) -> Transfer learning feature extraction.'

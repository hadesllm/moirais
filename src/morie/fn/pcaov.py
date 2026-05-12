# morie.fn -- function file (hadesllm/morie)
"""PCA on overlapping signal windows."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def pca_overlap(x, window: int = 256, hop: int = 128, n_components: int = 3, **kwargs) -> DescriptiveResult:
    """PCA on overlapping windows of a 1-D signal.

    Segments signal into overlapping frames, applies PCA, and returns
    the principal components of the windowed signal matrix.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    window : int
        Window length (default 256).
    hop : int
        Hop size (default 128).
    n_components : int
        Number of PCA components (default 3).

    Returns
    -------
    DescriptiveResult
        ``value`` is explained variance ratio of first component; ``extra``
        has ``components``, ``explained_variance_ratio``, ``scores``,
        ``n_frames``.

    References
    ----------
    Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.).
    Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    starts = list(range(0, N - window + 1, hop))
    if not starts:
        raise ValueError("Signal too short for given window size.")
    frames = np.array([x[s : s + window] for s in starts])
    n_frames = len(frames)
    mean = frames.mean(axis=0)
    centered = frames - mean
    cov = centered.T @ centered / (n_frames - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    k = min(n_components, len(eigvals))
    total_var = np.sum(eigvals) + 1e-15
    ratio = eigvals[:k] / total_var
    components = eigvecs[:, :k]
    scores = centered @ components
    return DescriptiveResult(
        name="pca_overlap",
        value=float(ratio[0]),
        extra={"components": components, "explained_variance_ratio": ratio, "scores": scores, "n_frames": n_frames},
    )


pcaov = pca_overlap


def cheatsheet() -> str:
    return "pca_overlap({}) -> PCA on overlapping signal windows."

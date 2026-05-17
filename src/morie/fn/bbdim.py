# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Blackbox dimensionality selection via scree analysis."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bb_dimensionality_select(Z, max_dims: int = 5) -> DescriptiveResult:
    """Select optimal Blackbox dimensions via scree/elbow.

    :param Z: Respondent x issue data matrix.
    :param max_dims: Maximum dimensions to consider.
    :return: DescriptiveResult with suggested dimensionality.

    .. epigraph:: We must know. We will know. -- David Hilbert
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    Zc = Z - Z.mean(axis=0)
    S = np.linalg.svd(Zc, compute_uv=False)
    eigvals = S**2 / (Z.shape[0] - 1)
    k = min(max_dims, len(eigvals))
    eigvals = eigvals[:k]
    diffs = np.diff(eigvals)
    if len(diffs) > 0:
        elbow = int(np.argmax(np.abs(diffs))) + 1
    else:
        elbow = 1
    return DescriptiveResult(
        name="bb_dimensionality_select",
        value=elbow,
        extra={"eigenvalues": eigvals.tolist(), "suggested_dims": elbow, "max_dims": max_dims},
    )


bbdim = bb_dimensionality_select


def cheatsheet() -> str:
    return "bb_dimensionality_select({}) -> Blackbox dimensionality selection via scree analysis."

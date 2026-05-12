# morie.fn -- function file (hadesllm/morie)
"""Dimensionality test via eigenvalue analysis."""

from __future__ import annotations

from ._containers import DescriptiveResult


def dimensionality_test(data, max_dims=5) -> DescriptiveResult:
    """Test dimensionality via eigenvalue decomposition.

    .. epigraph:: "Ours is the fury." -- Baratheon, Game of Thrones
    """
    import numpy as np

    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    corr = np.corrcoef(X.T)
    eigenvalues = np.linalg.eigvalsh(corr)[::-1]
    max_dims = min(max_dims, len(eigenvalues))
    test_stats = []
    for d in range(1, max_dims + 1):
        explained = float(np.sum(eigenvalues[:d]) / np.sum(eigenvalues))
        test_stats.append(
            {
                "dims": d,
                "explained_var": explained,
                "eigenvalue": float(eigenvalues[d - 1]) if d <= len(eigenvalues) else 0.0,
            }
        )
    return DescriptiveResult(
        name="dimensionality_test",
        value=float(eigenvalues[0]),
        extra={
            "test_stats": test_stats,
            "eigenvalues": eigenvalues[:max_dims].tolist(),
            "max_dims": max_dims,
        },
    )


dimts = dimensionality_test


def cheatsheet() -> str:
    return "dimensionality_test({}) -> Dimensionality test via eigenvalue analysis."

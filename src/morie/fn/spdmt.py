"""Sparse diagonal matrix operations."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def sparse_diagonal(
    diags: list | np.ndarray,
    *,
    offsets: list | None = None,
    n: int | None = None,
) -> DescriptiveResult:
    """We are what we repeatedly do. Excellence is a habit. — Aristotle"""
    if offsets is None:
        offsets = [0]
    if not isinstance(diags[0], np.ndarray):
        diags = [np.asarray(d, dtype=float) for d in diags]
    if n is None:
        main_idx = offsets.index(0) if 0 in offsets else 0
        n = len(diags[main_idx])
    M = np.zeros((n, n))
    for diag, off in zip(diags, offsets):
        for i in range(len(diag)):
            r = i - min(off, 0)
            c = i + max(off, 0)
            if 0 <= r < n and 0 <= c < n:
                M[r, c] = diag[i]
    nnz = int(np.count_nonzero(M))
    return DescriptiveResult(name="SparseDiagonal", value=nnz, extra={"matrix": M})


spdmt = sparse_diagonal

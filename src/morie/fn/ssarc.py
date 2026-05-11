"""Singular Spectrum Analysis reconstruction from selected components."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In a dark place we find ourselves, and a little more knowledge lights our way."


def ssa_reconstruct_fn(
    x: np.ndarray,
    L: int | None = None,
    groups: list | None = None,
) -> DescriptiveResult:
    """Reconstruct signal from selected SSA components.

    :param x: 1-D input signal.
    :param L: Window length (default N//2).
    :param groups: List of component indices to include (default [0]).
    :return: DescriptiveResult with reconstructed signal.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    if L is None:
        L = N // 2
    if groups is None:
        groups = [0]
    K = N - L + 1
    X = np.zeros((L, K))
    for i in range(K):
        X[:, i] = x[i : i + L]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    reconstructed = np.zeros(N)
    for idx in groups:
        if idx >= len(s):
            continue
        Xi = s[idx] * np.outer(U[:, idx], Vt[idx, :])
        rc = np.zeros(N)
        counts = np.zeros(N)
        for j in range(L):
            for k in range(K):
                rc[j + k] += Xi[j, k]
                counts[j + k] += 1
        rc /= counts
        reconstructed += rc
    residual = x - reconstructed
    return DescriptiveResult(
        name="ssa_reconstruct",
        value=float(np.mean(residual**2)),
        extra={
            "reconstructed": reconstructed,
            "residual": residual,
            "groups": groups,
            "L": L,
        },
    )


ssarc = ssa_reconstruct_fn


def cheatsheet() -> str:
    return "ssa_reconstruct_fn({}) -> Singular Spectrum Analysis reconstruction from selected comp"

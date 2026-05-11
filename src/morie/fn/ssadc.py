"""Singular Spectrum Analysis decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Truly wonderful, the mind of a child is."


def ssa_decompose_fn(x: np.ndarray, L: int | None = None) -> DescriptiveResult:
    """Decompose signal via Singular Spectrum Analysis (SSA).

    Embeds the signal into a trajectory (Hankel) matrix, performs SVD,
    and returns singular values and component matrices.

    :param x: 1-D input signal.
    :param L: Window length (default N//2).
    :return: DescriptiveResult with singular values and components.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    if L is None:
        L = N // 2
    K = N - L + 1
    X = np.zeros((L, K))
    for i in range(K):
        X[:, i] = x[i : i + L]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    components = []
    for i in range(len(s)):
        Xi = s[i] * np.outer(U[:, i], Vt[i, :])
        rc = np.zeros(N)
        counts = np.zeros(N)
        for j in range(L):
            for k in range(K):
                rc[j + k] += Xi[j, k]
                counts[j + k] += 1
        rc /= counts
        components.append(rc)
    return DescriptiveResult(
        name="ssa_decompose",
        value=float(len(s)),
        extra={
            "singular_values": s,
            "components": np.array(components),
            "L": L,
            "K": K,
            "n_components": len(s),
        },
    )


ssadc = ssa_decompose_fn


def cheatsheet() -> str:
    return "ssa_decompose_fn({}) -> Singular Spectrum Analysis decomposition."

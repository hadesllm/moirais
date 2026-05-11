"""Signal/noise subspace separation via SVD."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You must unlearn what you have learned."


def subspace_decompose_fn(x: np.ndarray, dim: int | None = None) -> DescriptiveResult:
    """Separate signal and noise subspaces via SVD of the data matrix.

    :param x: 1-D input signal.
    :param dim: Signal subspace dimension (default: estimated from eigenvalue gap).
    :return: DescriptiveResult with signal/noise subspace bases and singular values.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    L = N // 2
    K = N - L + 1
    X = np.zeros((L, K))
    for i in range(K):
        X[:, i] = x[i : i + L]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if dim is None:
        ratios = s[:-1] / s[1:]
        dim = int(np.argmax(ratios)) + 1
        dim = max(1, min(dim, len(s) - 1))
    signal_basis = U[:, :dim]
    noise_basis = U[:, dim:]
    return DescriptiveResult(
        name="subspace_decompose",
        value=float(dim),
        extra={
            "signal_basis": signal_basis,
            "noise_basis": noise_basis,
            "singular_values": s,
            "signal_dim": dim,
        },
    )


svspc = subspace_decompose_fn


def cheatsheet() -> str:
    return "subspace_decompose_fn({}) -> Signal/noise subspace separation via SVD."

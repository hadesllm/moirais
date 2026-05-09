"""Walsh-Hadamard Transform (1/sqrt(d) normalization)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def walsh_hadamard(
    x: np.ndarray,
    inverse: bool = False,
) -> DescriptiveResult:
    """Walsh-Hadamard Transform with 1/sqrt(d) normalization.

    CRITICAL: Uses 1/sqrt(d) normalization so that WHT is its own
    inverse (orthonormal). Using 1/d makes the inverse transform fail.

    :param x: Input vector. Padded to next power of 2 if needed.
    :param inverse: If True, apply inverse WHT (same operation due to symmetry).
    :return: DescriptiveResult with transformed vector.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    d_orig = len(x)
    n = int(2 ** np.ceil(np.log2(max(d_orig, 2))))
    if n > d_orig:
        x = np.concatenate([x, np.zeros(n - d_orig)])
    h = x.copy()
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(step):
                a, b = h[i + j], h[i + j + step]
                h[i + j] = a + b
                h[i + j + step] = a - b
        step *= 2
    h *= 1.0 / np.sqrt(n)
    return DescriptiveResult(
        name="walsh_hadamard",
        value=float(np.linalg.norm(h[:d_orig])),
        extra={
            "transformed": h[:d_orig],
            "full": h,
            "d_original": d_orig,
            "d_padded": n,
            "normalization": "1/sqrt(d)",
        },
    )


def cheatsheet() -> str:
    return "walsh_hadamard(x) -> WHT with 1/sqrt(d) normalization"


wht = walsh_hadamard

# morie.fn -- function file (hadesllm/morie)
"""Generate synthetic MA process."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Stay on target."


def ma_generate_fn(
    ma_coeffs: np.ndarray,
    sigma2: float = 1.0,
    N: int = 1000,
    seed: int | None = None,
) -> DescriptiveResult:
    r"""Generate a realisation of an MA(q) process.

    .. math::

        x(n) = w(n) + \\sum_{k=1}^{q} b_k w(n-k), \\quad w \\sim \\mathcal{N}(0, \\sigma^2)

    :param ma_coeffs: MA coefficients [b1, ..., bq].
    :param sigma2: Innovation variance (default 1.0).
    :param N: Number of samples to generate (default 1000).
    :param seed: Random seed (default None).
    :return: DescriptiveResult with generated signal.
    """
    ma_coeffs = np.asarray(ma_coeffs, dtype=float).ravel()
    q = len(ma_coeffs)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, np.sqrt(sigma2), N + q)
    x = np.zeros(N)
    for i in range(N):
        x[i] = w[i + q]
        for k in range(q):
            x[i] += ma_coeffs[k] * w[i + q - k - 1]
    return DescriptiveResult(
        name="ma_generate",
        value=float(N),
        extra={"signal": x, "ma_coeffs": ma_coeffs, "sigma2": sigma2, "N": N},
    )


magen = ma_generate_fn


def cheatsheet() -> str:
    return "ma_generate_fn({}) -> Generate synthetic MA process."

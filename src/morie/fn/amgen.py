# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Generate synthetic ARMA process."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "This is the way."


def arma_generate_fn(
    ar_coeffs: np.ndarray,
    ma_coeffs: np.ndarray,
    sigma2: float = 1.0,
    N: int = 1000,
    seed: int | None = None,
) -> DescriptiveResult:
    r"""Generate a realisation of an ARMA(p,q) process.

    .. math::

        x(n) = \\sum_{k=1}^{p} a_k x(n-k) + w(n) + \\sum_{k=1}^{q} b_k w(n-k)

    :param ar_coeffs: AR coefficients [a1, ..., ap].
    :param ma_coeffs: MA coefficients [b1, ..., bq].
    :param sigma2: Innovation variance (default 1.0).
    :param N: Number of samples to generate (default 1000).
    :param seed: Random seed (default None).
    :return: DescriptiveResult with generated signal.
    """
    ar_coeffs = np.asarray(ar_coeffs, dtype=float).ravel()
    ma_coeffs = np.asarray(ma_coeffs, dtype=float).ravel()
    p = len(ar_coeffs)
    q = len(ma_coeffs)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, np.sqrt(sigma2), N)
    x = np.zeros(N)
    for i in range(N):
        x[i] = w[i]
        for k in range(min(i, p)):
            x[i] += ar_coeffs[k] * x[i - k - 1]
        for k in range(min(i, q)):
            x[i] += ma_coeffs[k] * w[i - k - 1]
    return DescriptiveResult(
        name="arma_generate",
        value=float(N),
        extra={
            "signal": x,
            "ar_coeffs": ar_coeffs,
            "ma_coeffs": ma_coeffs,
            "sigma2": sigma2,
            "N": N,
        },
    )


amgen = arma_generate_fn


def cheatsheet() -> str:
    return "arma_generate_fn({}) -> Generate synthetic ARMA process."

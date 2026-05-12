# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Generate synthetic AR process."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Confine yourself to the present. -- Marcus Aurelius"


def ar_generate_fn(
    ar_coeffs: np.ndarray,
    sigma2: float = 1.0,
    N: int = 1000,
    seed: int | None = None,
) -> DescriptiveResult:
    r"""Generate a realisation of an AR(p) process.

    .. math::

        x(n) = \\sum_{k=1}^{p} a_k x(n-k) + w(n), \\quad w \\sim \\mathcal{N}(0, \\sigma^2)

    :param ar_coeffs: AR coefficients [a1, ..., ap].
    :param sigma2: Driving noise variance (default 1.0).
    :param N: Number of samples to generate (default 1000).
    :param seed: Random seed (default None).
    :return: DescriptiveResult with generated signal.
    """
    ar_coeffs = np.asarray(ar_coeffs, dtype=float).ravel()
    p = len(ar_coeffs)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, np.sqrt(sigma2), N)
    x = np.zeros(N)
    for i in range(N):
        x[i] = w[i]
        for k in range(min(i, p)):
            x[i] += ar_coeffs[k] * x[i - k - 1]
    return DescriptiveResult(
        name="ar_generate",
        value=float(N),
        extra={"signal": x, "ar_coeffs": ar_coeffs, "sigma2": sigma2, "N": N},
    )


argen = ar_generate_fn


def cheatsheet() -> str:
    return "ar_generate_fn({}) -> Generate synthetic AR process."

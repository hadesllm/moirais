# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""ARMA estimation via Newton-Raphson gradient descent."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def arma_newton_raphson_fn(
    x: np.ndarray,
    p: int = 4,
    q: int = 2,
    n_iter: int = 20,
) -> DescriptiveResult:
    """Estimate ARMA(p,q) coefficients using Newton-Raphson iteration.

    :param x: 1-D input signal.
    :param p: AR order (default 4).
    :param q: MA order (default 2).
    :param n_iter: Number of gradient iterations (default 20).
    :return: DescriptiveResult with AR/MA coefficients and residual variance.
    """
    from moirais._biomodel import arma_newton_raphson

    x = np.asarray(x, dtype=float).ravel()
    a, b, sigma2 = arma_newton_raphson(x, p=p, q=q, n_iter=n_iter)
    return DescriptiveResult(
        name="arma_newton_raphson",
        value=sigma2,
        extra={"ar_coeffs": a, "ma_coeffs": b, "sigma2": sigma2, "p": p, "q": q},
    )


armnr = arma_newton_raphson_fn


def cheatsheet() -> str:
    return "arma_newton_raphson_fn({}) -> ARMA estimation via Newton-Raphson gradient descent."

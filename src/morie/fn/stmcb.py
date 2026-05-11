"""ARMA estimation via Steiglitz-McBride iteration."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def steiglitz_mcbride_fn(
    x: np.ndarray,
    p: int = 4,
    q: int = 2,
    n_iter: int = 10,
) -> DescriptiveResult:
    """Estimate ARMA coefficients using Steiglitz-McBride method.

    :param x: 1-D input signal.
    :param p: AR order (default 4).
    :param q: MA order (default 2).
    :param n_iter: Number of iterations (default 10).
    :return: DescriptiveResult with AR/MA coefficients and residual variance.
    """
    from morie._biomodel import steiglitz_mcbride

    x = np.asarray(x, dtype=float).ravel()
    a, b, sigma2 = steiglitz_mcbride(x, p=p, q=q, n_iter=n_iter)
    return DescriptiveResult(
        name="steiglitz_mcbride",
        value=sigma2,
        extra={"ar_coeffs": a, "ma_coeffs": b, "sigma2": sigma2},
    )


stmcb = steiglitz_mcbride_fn


def cheatsheet() -> str:
    return "steiglitz_mcbride_fn({}) -> ARMA estimation via Steiglitz-McBride iteration."

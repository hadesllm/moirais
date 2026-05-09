# moirais.fn — function file (hadesllm/moirais)
"""ARMA estimation via modified Yule-Walker equations."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def modified_yule_walker_arma_fn(
    x: np.ndarray,
    p: int = 4,
    q: int = 2,
) -> DescriptiveResult:
    """Estimate ARMA AR coefficients using modified Yule-Walker method.

    :param x: 1-D input signal.
    :param p: AR order (default 4).
    :param q: MA order (default 2).
    :return: DescriptiveResult with AR coefficients and residual variance.
    """
    from moirais._biomodel import modified_yule_walker_arma

    x = np.asarray(x, dtype=float).ravel()
    a, sigma2 = modified_yule_walker_arma(x, p=p, q=q)
    return DescriptiveResult(
        name="modified_yw_arma",
        value=sigma2,
        extra={"ar_coeffs": a, "sigma2": sigma2, "p": p, "q": q},
    )


mywkr = modified_yule_walker_arma_fn


def cheatsheet() -> str:
    return "modified_yule_walker_arma_fn({}) -> ARMA estimation via modified Yule-Walker equations."

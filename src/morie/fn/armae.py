# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""ARMA coefficient estimation via modified Yule-Walker."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies -- with thunderous applause."


def arma_estimate_fn(x: np.ndarray, p: int = 4, q: int = 2) -> DescriptiveResult:
    """Estimate ARMA(p,q) coefficients via modified Yule-Walker equations.

    The AR part is estimated from high-lag autocorrelations (beyond lag q)
    to avoid MA contamination, then MA coefficients are derived.

    :param x: 1-D input signal.
    :param p: AR order (default 4).
    :param q: MA order (default 2).
    :return: DescriptiveResult with AR/MA coefficients and residual variance.
    """
    from morie._biomodel import modified_yule_walker_arma

    x = np.asarray(x, dtype=float).ravel()
    a, sigma2 = modified_yule_walker_arma(x, p=p, q=q)
    return DescriptiveResult(
        name="arma_estimate",
        value=float(sigma2),
        extra={"ar_coeffs": a, "sigma2": sigma2, "p": p, "q": q},
    )


armae = arma_estimate_fn


def cheatsheet() -> str:
    return "arma_estimate_fn({}) -> ARMA coefficient estimation via modified Yule-Walker."

# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayes factor (BIC approximation)."""

from __future__ import annotations

import numpy as np

from ._containers import ESRes


def bayes_factor(
    loglik_1: float,
    loglik_0: float,
    k_1: int,
    k_0: int,
    n: int,
) -> ESRes:
    r"""
    Approximate Bayes factor using the BIC (Schwarz criterion).

    .. math::

        BF_{10} \\approx \\exp\\left(\\frac{BIC_0 - BIC_1}{2}\\right)

    Parameters
    ----------
    loglik_1 : float
        Log-likelihood of model 1 (alternative).
    loglik_0 : float
        Log-likelihood of model 0 (null).
    k_1, k_0 : int
        Number of parameters in each model.
    n : int
        Sample size.

    Returns
    -------
    ESRes
        estimate = BF10, extra has 'bic_0', 'bic_1', 'interpretation'.

    References
    ----------
    Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *JASA*,
    90(430), 773-795.
    """
    if n <= 0:
        raise ValueError("n must be positive.")

    bic_1 = -2 * loglik_1 + k_1 * np.log(n)
    bic_0 = -2 * loglik_0 + k_0 * np.log(n)
    bf10 = np.exp((bic_0 - bic_1) / 2)

    if bf10 > 100:
        interp = "decisive"
    elif bf10 > 10:
        interp = "strong"
    elif bf10 > 3:
        interp = "substantial"
    elif bf10 > 1:
        interp = "barely worth mentioning"
    else:
        interp = "supports null"

    return ESRes(
        measure="bayes_factor",
        estimate=float(bf10),
        n=n,
        extra={
            "bic_0": float(bic_0),
            "bic_1": float(bic_1),
            "interpretation": interp,
        },
    )


bbf = bayes_factor


def cheatsheet() -> str:
    return "bayes_factor({}) -> Bayes factor (BIC approximation)."

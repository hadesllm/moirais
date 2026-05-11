# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian Information Criterion for AR model order selection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were my brother, Anakin. I loved you."


def bic_score_fn(x: np.ndarray, max_order: int = 20) -> DescriptiveResult:
    """Compute BIC for AR model orders 1..max_order.

    .. math::

        \\text{BIC}(p) = N \\ln(\\hat{\\sigma}^2_p) + p \\ln(N)

    :param x: 1-D input signal.
    :param max_order: Maximum AR order to evaluate (default 20).
    :return: DescriptiveResult with optimal order and BIC scores.
    """
    from morie._armodel import optimal_ar_order

    x = np.asarray(x, dtype=float).ravel()
    best_order, scores = optimal_ar_order(x, max_order=max_order, criterion="bic")
    return DescriptiveResult(
        name="bic_score",
        value=float(best_order),
        extra={"best_order": best_order, "scores": scores, "criterion": "bic"},
    )


bicsc = bic_score_fn


def cheatsheet() -> str:
    return "bic_score_fn({}) -> Bayesian Information Criterion for AR model order selection."

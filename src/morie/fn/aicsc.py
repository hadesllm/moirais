# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Akaike Information Criterion for AR model order selection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The ability to speak does not make you intelligent."


def aic_score_fn(x: np.ndarray, max_order: int = 20) -> DescriptiveResult:
    r"""Compute AIC for AR model orders 1..max_order.

    .. math::

        \\text{AIC}(p) = N \\ln(\\hat{\\sigma}^2_p) + 2p

    :param x: 1-D input signal.
    :param max_order: Maximum AR order to evaluate (default 20).
    :return: DescriptiveResult with optimal order and AIC scores.
    """
    from morie._armodel import optimal_ar_order

    x = np.asarray(x, dtype=float).ravel()
    best_order, scores = optimal_ar_order(x, max_order=max_order, criterion="aic")
    return DescriptiveResult(
        name="aic_score",
        value=float(best_order),
        extra={"best_order": best_order, "scores": scores, "criterion": "aic"},
    )


aicsc = aic_score_fn


def cheatsheet() -> str:
    return "aic_score_fn({}) -> Akaike Information Criterion for AR model order selection."

# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Optimal AR model order selection via information criterion."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ar_order_select(
    x: np.ndarray,
    max_order: int = 30,
    criterion: str = "aic",
) -> DescriptiveResult:
    """Select optimal AR model order using AIC, BIC, or MDL.

    :param x: 1-D input signal.
    :param max_order: Maximum order to test (default 30).
    :param criterion: Information criterion: 'aic', 'bic', or 'mdl' (default 'aic').
    :return: DescriptiveResult with best_order as value and scores in extra.
    """
    from morie._armodel import optimal_ar_order

    x = np.asarray(x, dtype=float).ravel()
    best_order, scores = optimal_ar_order(x, max_order=max_order, criterion=criterion)
    return DescriptiveResult(
        name="ar_order_select",
        value=int(best_order),
        extra={"scores": scores, "criterion": criterion},
    )


arord = ar_order_select


def cheatsheet() -> str:
    return "ar_order_select({}) -> Optimal AR model order selection via information criterion."

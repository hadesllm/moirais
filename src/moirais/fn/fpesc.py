# moirais.fn — function file (hadesllm/moirais)
"""Final Prediction Error for AR model order selection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "A long time ago in a galaxy far, far away."


def fpe_score_fn(x: np.ndarray, max_order: int = 20) -> DescriptiveResult:
    """Compute Akaike's Final Prediction Error for AR orders 1..max_order.

    .. math::

        \\text{FPE}(p) = \\hat{\\sigma}^2_p \\cdot \\frac{N + p + 1}{N - p - 1}

    :param x: 1-D input signal.
    :param max_order: Maximum AR order to evaluate (default 20).
    :return: DescriptiveResult with optimal order and FPE scores.
    """
    from moirais._armodel import optimal_ar_order

    x = np.asarray(x, dtype=float).ravel()
    best_order, scores = optimal_ar_order(x, max_order=max_order, criterion="fpe")
    return DescriptiveResult(
        name="fpe_score",
        value=float(best_order),
        extra={"best_order": best_order, "scores": scores, "criterion": "fpe"},
    )


fpesc = fpe_score_fn


def cheatsheet() -> str:
    return "fpe_score_fn({}) -> Final Prediction Error for AR model order selection."

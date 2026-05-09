# moirais.fn — function file (hadesllm/moirais)
"""Minimum Description Length for AR model order selection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is always a bigger fish."


def mdl_score_fn(x: np.ndarray, max_order: int = 20) -> DescriptiveResult:
    """Compute MDL (Rissanen) for AR model orders 1..max_order.

    .. math::

        \\text{MDL}(p) = N \\ln(\\hat{\\sigma}^2_p) + p \\ln(N)

    MDL and BIC share the same formula but differ in derivation
    (information-theoretic vs Bayesian).

    :param x: 1-D input signal.
    :param max_order: Maximum AR order to evaluate (default 20).
    :return: DescriptiveResult with optimal order and MDL scores.
    """
    from moirais._armodel import ar_burg

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    scores = np.zeros(max_order)
    for p in range(1, max_order + 1):
        _, sigma2 = ar_burg(x, p)
        sigma2 = max(sigma2, 1e-20)
        scores[p - 1] = n * np.log(sigma2) + 0.5 * p * np.log(n)
    best_order = int(np.argmin(scores)) + 1
    return DescriptiveResult(
        name="mdl_score",
        value=float(best_order),
        extra={"best_order": best_order, "scores": scores, "criterion": "mdl"},
    )


mdlsc = mdl_score_fn


def cheatsheet() -> str:
    return "mdl_score_fn({}) -> Minimum Description Length for AR model order selection."

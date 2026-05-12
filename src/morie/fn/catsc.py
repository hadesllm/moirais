# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Criterion Autoregressive Transfer function for model order selection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I've got a bad feeling about this."


def cat_score_fn(x: np.ndarray, max_order: int = 20) -> DescriptiveResult:
    r"""Compute CAT (Parzen) criterion for AR orders 1..max_order.

    .. math::

        \\text{CAT}(p) = \\frac{1}{N} \\sum_{k=1}^{p} \\frac{1}{\\hat{\\sigma}^2_k}
        - \\frac{1}{\\hat{\\sigma}^2_p}

    :param x: 1-D input signal.
    :param max_order: Maximum AR order to evaluate (default 20).
    :return: DescriptiveResult with optimal order and CAT scores.
    """
    from morie._armodel import ar_burg

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    sigmas = np.zeros(max_order)
    for p in range(1, max_order + 1):
        _, sigma2 = ar_burg(x, p)
        sigmas[p - 1] = max(sigma2, 1e-20)
    scores = np.zeros(max_order)
    for p in range(1, max_order + 1):
        inv_sum = sum(1.0 / sigmas[k] for k in range(p))
        scores[p - 1] = inv_sum / n - 1.0 / sigmas[p - 1]
    best_order = int(np.argmin(scores)) + 1
    return DescriptiveResult(
        name="cat_score",
        value=float(best_order),
        extra={"best_order": best_order, "scores": scores, "criterion": "cat"},
    )


catsc = cat_score_fn


def cheatsheet() -> str:
    return "cat_score_fn({}) -> Criterion Autoregressive Transfer function for model order s"

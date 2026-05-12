"""L2 weight decay."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def weight_decay(
    params: list[np.ndarray],
    lambda_: float = 0.01,
) -> DescriptiveResult:
    r"""Apply L2 weight decay (penalty) to parameter arrays.

    Computes the L2 penalty :math:`\\frac{\\lambda}{2} \\sum \\|w\\|^2` and
    returns the decayed parameters :math:`w \\leftarrow w (1 - \\lambda)`.

    :param params: List of parameter arrays.
    :param lambda_: Weight decay coefficient.
    :return: DescriptiveResult with decayed params and penalty.
    """
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be >= 0, got {lambda_}")

    penalty = 0.5 * lambda_ * sum(float(np.sum(p**2)) for p in params)
    decayed = [p * (1.0 - lambda_) for p in params]

    return DescriptiveResult(
        name="weight_decay",
        value=float(penalty),
        extra={"decayed": decayed, "lambda": lambda_, "n_params": len(params)},
    )


def cheatsheet() -> str:
    return "weight_decay(params, lambda_) -> L2 penalty and decayed params"


wdecm = weight_decay

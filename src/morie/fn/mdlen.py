# morie.fn — function file (hadesllm/morie)
"""Minimum description length (MDL)."""

import numpy as np

from ._containers import ESRes


def mdl(log_likelihood: float, k: int, n: int, **kwargs) -> ESRes:
    r"""
    Compute minimum description length criterion.

    .. math::

        MDL = -\\ln(L) + \\frac{k}{2} \\ln(n)

    Equivalent to BIC / 2. Smaller values indicate better model.

    :param log_likelihood: Log-likelihood of the model.
    :param k: Number of estimated parameters.
    :param n: Sample size.
    :return: ESRes with MDL value.

    References
    ----------
    Rissanen J (1978). Modeling by shortest data description.
    Automatica, 14(5), 465-471.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    mdl_val = -log_likelihood + 0.5 * k * np.log(n)
    return ESRes(
        measure="mdl",
        estimate=float(mdl_val),
        n=n,
        extra={"k": k, "log_likelihood": log_likelihood, "BIC": float(2 * mdl_val)},
    )


mdlen = mdl


def cheatsheet() -> str:
    return "mdl(log_likelihood, k, n) -> Minimum description length."

# moirais.fn — function file (hadesllm/moirais)
"""Cramer-Rao bound."""

import numpy as np

from ._containers import ESRes


def cramer_rao_bound(fisher_info: float, n: int = 1, **kwargs) -> ESRes:
    """
    Compute Cramer-Rao lower bound on estimator variance.

    .. math::

        \\text{Var}(\\hat{\\theta}) \\geq \\frac{1}{n \\cdot I(\\theta)}

    :param fisher_info: Fisher information I(theta) for a single observation.
    :param n: Sample size.
    :return: ESRes with the Cramer-Rao lower bound.

    References
    ----------
    Cramer H (1946). Mathematical Methods of Statistics. Princeton.
    Rao CR (1945). Information and the accuracy attainable in the
    estimation of statistical parameters. Bull Calcutta Math Soc, 37, 81-91.
    """
    if fisher_info <= 0:
        raise ValueError("Fisher information must be positive.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    crlb = 1.0 / (n * fisher_info)
    return ESRes(
        measure="cramer_rao_bound",
        estimate=crlb,
        n=n,
        extra={"fisher_info": fisher_info, "min_se": float(np.sqrt(crlb))},
    )


cramr = cramer_rao_bound


def cheatsheet() -> str:
    return "cramer_rao_bound(fisher_info, n) -> Cramer-Rao lower bound."

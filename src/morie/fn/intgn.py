# morie.fn -- function file (hadesllm/morie)
"""Integration (neural complexity measure)."""

import numpy as np

from ._containers import ESRes


def integration(cov_matrix, **kwargs) -> ESRes:
    r"""
    Compute integration (neural complexity) from a covariance matrix.

    Integration I(X) = sum of marginal entropies - joint entropy.
    For Gaussian variables:

    .. math::

        I(X) = \\frac{1}{2}\\sum_i \\ln(\\sigma_i^2) - \\frac{1}{2}\\ln|\\Sigma|

    :param cov_matrix: (p, p) covariance matrix.
    :return: ESRes with integration in nats.

    References
    ----------
    Tononi G, Sporns O, Edelman GM (1994). A measure for brain
    complexity: relating functional segregation and integration in
    the nervous system. PNAS, 91(11), 5033-5037.
    """
    C = np.asarray(cov_matrix, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")
    p = C.shape[0]
    if p < 2:
        raise ValueError("Need at least 2 variables.")

    variances = np.diag(C)
    if np.any(variances <= 0):
        raise ValueError("Diagonal of covariance must be positive.")

    sum_marginal = 0.5 * float(np.sum(np.log(variances)))
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")
    joint = 0.5 * logdet
    integ = sum_marginal - joint

    return ESRes(
        measure="integration",
        estimate=float(integ),
        n=p,
        extra={"n_variables": p, "log_det": logdet},
    )


intgn = integration


def cheatsheet() -> str:
    return "integration(cov_matrix) -> Neural complexity integration."

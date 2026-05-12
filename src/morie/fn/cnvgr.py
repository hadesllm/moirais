# morie.fn — function file (hadesllm/morie)
"""LMS convergence rate."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Much to learn, you still have."


def convergence_rate(mu: float, eigenvalues, **kwargs) -> DescriptiveResult:
    r"""Compute the LMS adaptive filter convergence time constants.

    .. math::

        \\tau_k = \\frac{1}{2 \\mu \\lambda_k}

    Parameters
    ----------
    mu : float
        Step size parameter.
    eigenvalues : array-like
        Eigenvalues of the input autocorrelation matrix.

    Returns
    -------
    DescriptiveResult
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if mu <= 0:
        raise ValueError("Step size mu must be positive.")
    if np.any(eigenvalues <= 0):
        raise ValueError("Eigenvalues must be positive.")
    tau = 1.0 / (2.0 * mu * eigenvalues)
    tau_max = float(np.max(tau))
    tau_min = float(np.min(tau))
    return DescriptiveResult(
        name="convergence_rate",
        value=tau_max,
        extra={
            "tau": tau,
            "tau_max": tau_max,
            "tau_min": tau_min,
            "mu": mu,
            "eigenvalues": eigenvalues,
        },
    )


cnvgr = convergence_rate


def cheatsheet() -> str:
    return "convergence_rate({}) -> LMS convergence rate."

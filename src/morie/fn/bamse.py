# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Posterior standard errors."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_se_from_posterior(chain) -> DescriptiveResult:
    """Standard errors from posterior chain.

    .. epigraph:: "Dracarys!" -- Daenerys, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float)
    if chain.ndim == 1:
        chain = chain.reshape(-1, 1)
    ses = np.std(chain, axis=0, ddof=1)
    return DescriptiveResult(
        name="bayesian_se_from_posterior",
        value=float(ses[0]),
        extra={
            "ses": ses.tolist(),
            "n_samples": chain.shape[0],
            "n_params": chain.shape[1],
        },
    )


bamse = bayesian_se_from_posterior


def cheatsheet() -> str:
    return "bayesian_se_from_posterior({}) -> Posterior standard errors."

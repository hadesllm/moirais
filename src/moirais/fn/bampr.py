# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian posterior summary."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bayesian_am_posterior_summary(chain) -> DescriptiveResult:
    """Compute posterior mean, sd, and credible intervals.

    .. epigraph:: "You know nothing." -- Ygritte, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float)
    if chain.ndim == 1:
        chain = chain.reshape(-1, 1)
    means = np.mean(chain, axis=0)
    sds = np.std(chain, axis=0, ddof=1)
    ci_lo = np.percentile(chain, 2.5, axis=0)
    ci_hi = np.percentile(chain, 97.5, axis=0)
    return DescriptiveResult(
        name="bayesian_am_posterior_summary",
        value=float(means[0]),
        extra={
            "means": means.tolist(),
            "sds": sds.tolist(),
            "ci_lo": ci_lo.tolist(),
            "ci_hi": ci_hi.tolist(),
            "n_samples": chain.shape[0],
            "n_params": chain.shape[1],
        },
    )


bampr = bayesian_am_posterior_summary


def cheatsheet() -> str:
    return "bayesian_am_posterior_summary({}) -> Bayesian posterior summary."

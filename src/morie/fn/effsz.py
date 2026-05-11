# morie.fn — function file (hadesllm/morie)
"""Effective sample size for MCMC."""

from __future__ import annotations

from ._containers import DescriptiveResult


def effective_sample_size(chain) -> DescriptiveResult:
    """ESS using initial monotone sequence estimator.

    .. epigraph:: "The North remembers." -- Various, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float).ravel()
    n = len(chain)
    mean = np.mean(chain)
    var = np.var(chain, ddof=1)
    if var < 1e-15:
        return DescriptiveResult(name="effective_sample_size", value=float(n), extra={"ess": float(n), "n": n})
    max_lag = min(n - 1, 200)
    acf_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = np.corrcoef(chain[: n - lag], chain[lag:])[0, 1]
        if rho < 0:
            break
        acf_sum += rho
    ess = n / (1.0 + 2.0 * acf_sum)
    return DescriptiveResult(
        name="effective_sample_size",
        value=float(ess),
        extra={"ess": float(ess), "n": n, "acf_sum": float(acf_sum)},
    )


effsz = effective_sample_size


def cheatsheet() -> str:
    return "effective_sample_size({}) -> Effective sample size for MCMC."

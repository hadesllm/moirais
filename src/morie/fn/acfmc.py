# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""MCMC autocorrelation function."""

from __future__ import annotations

from ._containers import DescriptiveResult


def autocorrelation_mcmc(chain, max_lag=50) -> DescriptiveResult:
    """Compute ACF for an MCMC chain up to max_lag.

    .. epigraph:: "Hold the door." -- Hodor, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float).ravel()
    n = len(chain)
    max_lag = min(max_lag, n - 1)
    mean = np.mean(chain)
    var = np.var(chain)
    if var < 1e-15:
        acf = np.ones(max_lag + 1)
    else:
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0
        for k in range(1, max_lag + 1):
            acf[k] = np.mean((chain[: n - k] - mean) * (chain[k:] - mean)) / var
    return DescriptiveResult(
        name="autocorrelation_mcmc",
        value=float(acf[1]) if max_lag >= 1 else 1.0,
        extra={"acf": acf.tolist(), "max_lag": max_lag, "n": n},
    )


acfmc = autocorrelation_mcmc


def cheatsheet() -> str:
    return "autocorrelation_mcmc({}) -> MCMC autocorrelation function."

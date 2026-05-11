# morie.fn — function file (hadesllm/morie)
"""Posterior predictive check."""

from __future__ import annotations

from ._containers import DescriptiveResult


def posterior_predictive_check(chain, data) -> DescriptiveResult:
    """Posterior predictive p-values.

    .. epigraph:: "Not today." -- Syrio, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float).ravel()
    data = np.asarray(data, dtype=float).ravel()
    obs_stat = float(np.mean(data))
    rng = np.random.default_rng(42)
    n_rep = min(len(chain), 500)
    rep_stats = []
    for i in range(n_rep):
        sim = rng.normal(chain[i], 1.0, size=len(data))
        rep_stats.append(float(np.mean(sim)))
    rep_stats = np.array(rep_stats)
    ppp = float(np.mean(rep_stats >= obs_stat))
    return DescriptiveResult(
        name="posterior_predictive_check",
        value=ppp,
        extra={
            "ppp_value": ppp,
            "observed_stat": obs_stat,
            "n_replications": n_rep,
        },
    )


pstpc = posterior_predictive_check


def cheatsheet() -> str:
    return "posterior_predictive_check({}) -> Posterior predictive check."

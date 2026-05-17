"""Monte Carlo spatial significance test with simulation envelopes."""

from __future__ import annotations

from ._containers import DescriptiveResult


def monte_carlo_spatial_test(Z, coords, stat_fn=None, n_sim=999):
    """Monte Carlo test for spatial pattern significance.

    Generates simulation envelopes under CSR/random relabeling.

    .. epigraph:: There is no royal road to geometry. -- Euclid

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    stat_fn : callable, optional
        Statistic function ``(Z, coords) -> float``. Defaults to variance.
    n_sim : int
        Number of simulations.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)

    if stat_fn is None:

        def stat_fn(z, c):
            return float(np.var(z))

    observed = float(stat_fn(Z, coords))

    rng = np.random.default_rng(42)
    sim_stats = np.empty(n_sim)
    for i in range(n_sim):
        Z_sim = rng.permutation(Z)
        sim_stats[i] = stat_fn(Z_sim, coords)

    lo = float(np.percentile(sim_stats, 2.5))
    hi = float(np.percentile(sim_stats, 97.5))
    p_value = float(np.mean(np.abs(sim_stats) >= np.abs(observed)))

    return DescriptiveResult(
        name="monte_carlo_spatial_test",
        value=observed,
        extra={
            "observed": observed,
            "envelope_lo": lo,
            "envelope_hi": hi,
            "p_value": p_value,
            "n_sim": n_sim,
            "significant": observed < lo or observed > hi,
        },
    )


sgmci = monte_carlo_spatial_test


def cheatsheet() -> str:
    return "monte_carlo_spatial_test({}) -> Monte Carlo spatial significance test with simulation envelo"

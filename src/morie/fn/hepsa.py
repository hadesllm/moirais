# morie.fn -- function file (hadesllm/morie)
"""Probabilistic sensitivity analysis for health economics."""

import numpy as np

from ._containers import DescriptiveResult


def probabilistic_sensitivity(
    param_distributions: dict[str, tuple[float, float]],
    n_sim: int = 1000,
    seed: int = 42,
) -> DescriptiveResult:
    """Run PSA by sampling parameters from Normal(mean, sd).

    Parameters
    ----------
    param_distributions : dict
        {name: (mean, sd)} for each uncertain parameter.
    n_sim : int
    seed : int

    Returns
    -------
    DescriptiveResult
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for name, (mu, sd) in param_distributions.items():
        samples[name] = rng.normal(mu, sd, n_sim).tolist()

    means = {k: float(np.mean(v)) for k, v in samples.items()}
    sds = {k: float(np.std(v)) for k, v in samples.items()}

    return DescriptiveResult(
        name="PSA",
        value=means,
        extra={"std_devs": sds, "n_sim": n_sim, "n_params": len(param_distributions)},
    )


hepsa = probabilistic_sensitivity


def cheatsheet() -> str:
    return "probabilistic_sensitivity({}) -> Probabilistic sensitivity analysis for health economics."

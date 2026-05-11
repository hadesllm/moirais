"""Simulate noisy perceptions of true positions."""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulate_perceptions(true_positions, n_resp: int = 50, sigma: float = 0.5, seed: int = 42) -> DescriptiveResult:
    """Generate noisy perceptual data from true stimulus positions.

    :param true_positions: True stimulus positions (n_stimuli,) or (n_stimuli, n_dims).
    :param n_resp: Number of respondents.
    :param sigma: Perception noise std.
    :param seed: Random seed.
    :return: DescriptiveResult with respondent x stimulus perception matrix.

    .. epigraph:: "Domain Expansion!" -- Gojo Satoru, Jujutsu Kaisen
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    true = np.asarray(true_positions, dtype=float)
    if true.ndim == 1:
        n_stim = len(true)
        perceptions = true[None, :] + rng.normal(0, sigma, (n_resp, n_stim))
    else:
        n_stim = true.shape[0]
        perceptions = np.tile(true, (n_resp, 1, 1)) + rng.normal(0, sigma, (n_resp, *true.shape))
    return DescriptiveResult(
        name="simulate_perceptions",
        value=n_resp,
        extra={"perceptions": perceptions.tolist(), "sigma": sigma, "n_stimuli": n_stim},
    )


simpe = simulate_perceptions


def cheatsheet() -> str:
    return "simulate_perceptions({}) -> Simulate noisy perceptions of true positions."

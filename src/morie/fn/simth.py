"""Simulate feeling thermometer data. 'Mastered Ultra Instinct.' -- Goku, Dragon Ball Z"""

from __future__ import annotations

from ._containers import DescriptiveResult


def simulate_thermometer(n_resp=50, n_stim=7, noise=5.0, seed=42):
    """Generate synthetic feeling thermometer data.

    Parameters
    ----------
    n_resp : int
        Number of respondents.
    n_stim : int
        Number of stimuli.
    noise : float
        Gaussian noise std.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        value = thermometer ratings (n_resp x n_stim), clipped to [0, 100].
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    ideal = rng.uniform(20, 80, (n_resp, 1))
    stim_pos = np.linspace(10, 90, n_stim)
    ratings = 100 - np.abs(ideal - stim_pos) + rng.normal(0, noise, (n_resp, n_stim))
    ratings = np.clip(ratings, 0, 100)
    return DescriptiveResult(
        name="simulate_thermometer",
        value=ratings,
        extra={"n_resp": n_resp, "n_stim": n_stim, "stim_positions": stim_pos.tolist()},
    )


simth = simulate_thermometer


def cheatsheet() -> str:
    return "simulate_thermometer({}) -> Simulate feeling thermometer data. 'Mastered Ultra Instinct."

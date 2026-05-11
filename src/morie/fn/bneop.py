# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Birth-death process. 'The prophecy was true.' -- Councillor Hamann"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def birth_death_process(
    *,
    birth_rate: float = 1.0,
    death_rate: float = 0.5,
    n0: int = 100,
    t_max: float = 10.0,
    dt: float = 0.01,
    seed: int | None = 42,
) -> DescriptiveResult:
    """Simulate a continuous-time birth-death (Markov) process.

    At each time step, the population changes by +1 with rate
    birth_rate * N and -1 with rate death_rate * N (Gillespie-like
    Euler approximation).

    Parameters
    ----------
    birth_rate : float
        Per-capita birth rate (lambda).
    death_rate : float
        Per-capita death rate (mu).
    n0 : int
        Initial population.
    t_max : float
        Simulation duration.
    dt : float
        Time step for Euler approximation.
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the final population; ``extra`` has the trajectory,
        extinction time (if applicable), and growth rate.

    References
    ----------
    Allen, L. J. S. (2003). An Introduction to Stochastic Processes with
    Applications to Biology. Pearson Prentice Hall.
    """
    if birth_rate < 0 or death_rate < 0:
        raise ValueError("Rates must be non-negative")
    if n0 < 0:
        raise ValueError("Initial population must be non-negative")

    rng = np.random.default_rng(seed)
    steps = int(t_max / dt)
    pop = np.zeros(steps + 1)
    pop[0] = n0
    extinction_time = None

    for i in range(steps):
        n = pop[i]
        if n <= 0:
            pop[i:] = 0
            extinction_time = i * dt
            break
        births = rng.poisson(birth_rate * n * dt)
        deaths = rng.poisson(death_rate * n * dt)
        pop[i + 1] = max(0, n + births - deaths)

    times = np.linspace(0, t_max, steps + 1)
    net_rate = birth_rate - death_rate
    expected_final = n0 * np.exp(net_rate * t_max)

    return DescriptiveResult(
        name="Birth-Death Process",
        value=float(pop[-1]),
        extra={
            "trajectory": pop[:: max(1, len(pop) // 50)].tolist(),
            "n0": n0,
            "birth_rate": birth_rate,
            "death_rate": death_rate,
            "net_growth_rate": net_rate,
            "expected_final": float(expected_final),
            "extinction_time": extinction_time,
            "t_max": t_max,
            "peak_population": float(pop.max()),
        },
    )


bneop = birth_death_process


def cheatsheet() -> str:
    return "birth_death_process({}) -> Birth-death process. 'The prophecy was true.' -- Councillor "

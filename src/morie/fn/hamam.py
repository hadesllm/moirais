# morie.fn — function file (hadesllm/morie)
"""Brownian motion simulation. 'I have survived your predecessors.' -- The Merovingian"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def brownian_motion(
    n_steps: int = 1000,
    *,
    n_paths: int = 1,
    dt: float = 0.01,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = 42,
) -> DescriptiveResult:
    """Simulate geometric or arithmetic Brownian motion paths.

    When mu=0, simulates standard Brownian motion (Wiener process).
    Otherwise simulates geometric Brownian motion:
    dS = mu*S*dt + sigma*S*dW.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of independent paths to simulate.
    dt : float
        Time increment.
    mu : float
        Drift coefficient.
    sigma : float
        Volatility (diffusion coefficient).
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the mean terminal value across paths; ``extra`` has
        paths array, times, and summary statistics.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    rng = np.random.default_rng(seed)
    dW = rng.normal(0, np.sqrt(dt), (n_paths, n_steps))

    if mu == 0.0:
        paths = np.cumsum(sigma * dW, axis=1)
        paths = np.column_stack([np.zeros(n_paths), paths])
    else:
        paths = np.ones((n_paths, n_steps + 1))
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, t])

    times = np.arange(n_steps + 1) * dt
    terminal = paths[:, -1]

    return DescriptiveResult(
        name="Brownian Motion",
        value=float(terminal.mean()),
        extra={
            "terminal_mean": float(terminal.mean()),
            "terminal_std": float(terminal.std()),
            "terminal_min": float(terminal.min()),
            "terminal_max": float(terminal.max()),
            "n_steps": n_steps,
            "n_paths": n_paths,
            "dt": dt,
            "mu": mu,
            "sigma": sigma,
            "T": float(times[-1]),
            "paths_shape": list(paths.shape),
        },
    )


hamam = brownian_motion


def cheatsheet() -> str:
    return "brownian_motion({}) -> Brownian motion simulation. 'I have survived your predecesso"

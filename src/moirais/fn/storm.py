"""Markov chain weather model. 'Do you know what happens to a toad when it is struck by lightning?' -- Storm"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def markov_weather(
    sequence: list[int] | np.ndarray,
    *,
    n_states: int | None = None,
    forecast_steps: int = 5,
    seed: int | None = None,
) -> DescriptiveResult:
    """Fit a first-order Markov chain and forecast future states.

    Estimates the transition matrix from an observed state sequence via
    maximum likelihood, computes the stationary distribution, and
    generates a probabilistic forecast.

    Parameters
    ----------
    sequence : array-like of int
        Observed state sequence (0-indexed).
    n_states : int or None
        Number of states. If None, inferred from data.
    forecast_steps : int
        Number of steps to forecast.
    seed : int or None
        RNG seed for forecast.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``transition_matrix``, ``stationary``,
        ``forecast`` (array), ``log_likelihood``.
    """
    seq = np.asarray(sequence, dtype=int)
    if seq.ndim != 1 or len(seq) < 2:
        raise ValueError("sequence must be 1D with at least 2 elements")

    if n_states is None:
        n_states = int(seq.max()) + 1
    if np.any(seq < 0) or np.any(seq >= n_states):
        raise ValueError(f"States must be in [0, {n_states - 1}]")

    T = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(seq) - 1):
        T[seq[i], seq[i + 1]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stat = np.real(eigvecs[:, idx])
    stat = stat / stat.sum()

    ll = 0.0
    for i in range(len(seq) - 1):
        p_ij = P[seq[i], seq[i + 1]]
        ll += np.log(max(p_ij, 1e-30))

    rng = np.random.default_rng(seed)
    forecast = np.empty(forecast_steps, dtype=int)
    state = seq[-1]
    for t in range(forecast_steps):
        probs = P[state]
        if probs.sum() < 1e-10:
            probs = np.ones(n_states) / n_states
        state = rng.choice(n_states, p=probs)
        forecast[t] = state

    return DescriptiveResult(
        name="markov_weather",
        value={
            "transition_matrix": P,
            "stationary": stat,
            "forecast": forecast,
            "log_likelihood": float(ll),
        },
        extra={"n_states": n_states, "n_obs": len(seq), "forecast_steps": forecast_steps},
    )


storm = markov_weather


def cheatsheet() -> str:
    return "markov_weather({}) -> Markov chain weather model. 'Do you know what happens to a t"

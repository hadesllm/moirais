# morie.fn -- function file (hadesllm/morie)
"""Stochastic sequence generation (Markov chain). 'Do I at least get to pick the song?' -- Jazz"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def markov_generate(
    transition_matrix: np.ndarray,
    *,
    n_steps: int = 100,
    start_state: int = 0,
    seed: int = 42,
) -> DescriptiveResult:
    """Generate a sequence from a discrete-time Markov chain.

    Parameters
    ----------
    transition_matrix : ndarray of shape (k, k)
        Row-stochastic transition matrix.
    n_steps : int
        Number of steps to generate.
    start_state : int
        Initial state index.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        With ``value`` = state sequence (ndarray of ints) and
        ``extra`` containing stationary distribution estimate.
    """
    P = np.asarray(transition_matrix, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square")
    k = P.shape[0]
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Rows must sum to 1 (row-stochastic)")
    if start_state < 0 or start_state >= k:
        raise ValueError(f"start_state must be in [0, {k - 1}]")

    rng = np.random.default_rng(seed)
    states = np.zeros(n_steps, dtype=int)
    states[0] = start_state
    for t in range(1, n_steps):
        states[t] = rng.choice(k, p=P[states[t - 1]])

    counts = np.bincount(states, minlength=k)
    empirical_stationary = counts / counts.sum()

    return DescriptiveResult(
        name="markov_chain_sequence",
        value=states,
        extra={"n_states": k, "n_steps": n_steps, "empirical_stationary": empirical_stationary},
    )


jazz = markov_generate


def cheatsheet() -> str:
    return "markov_generate({}) -> Stochastic sequence generation (Markov chain). 'Do I at leas"

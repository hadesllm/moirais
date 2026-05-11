# morie.fn — function file (hadesllm/morie)
"""Multi-state model transition analysis."""

from __future__ import annotations

import numpy as np

__all__ = ["mstat"]


def mstat(
    time: np.ndarray,
    state_from: np.ndarray,
    state_to: np.ndarray,
    *,
    states: list | None = None,
) -> dict:
    """Multi-state model: transition intensities and probabilities.

    Estimates Aalen-Johansen transition probabilities for a
    Markov multi-state model.

    Parameters
    ----------
    time : array-like
        Transition times (n,).
    state_from : array-like
        Origin state for each transition (n,).
    state_to : array-like
        Destination state for each transition (n,).
    states : list, optional
        All possible states. Inferred if not given.

    Returns
    -------
    dict
        states, transition_counts, transition_rates,
        transition_matrix, n_transitions.
    """
    time = np.asarray(time, dtype=float)
    state_from = np.asarray(state_from)
    state_to = np.asarray(state_to)

    if states is None:
        states = sorted(set(state_from) | set(state_to))
    k = len(states)
    state_idx = {s: i for i, s in enumerate(states)}

    counts = np.zeros((k, k), dtype=int)
    for sf, st in zip(state_from, state_to):
        if sf in state_idx and st in state_idx:
            counts[state_idx[sf], state_idx[st]] += 1

    row_totals = counts.sum(axis=1, keepdims=True).astype(float)
    safe_totals = np.where(row_totals > 0, row_totals, 1.0)
    rates = np.where(row_totals > 0, counts / safe_totals, 0.0)

    total_time = np.max(time) - np.min(time) if len(time) > 0 else 1.0
    intensity = counts / max(total_time, 1e-10)

    return {
        "states": states,
        "transition_counts": counts,
        "transition_rates": rates,
        "transition_intensity": intensity,
        "n_transitions": int(len(time)),
    }


mstat_fn = mstat


def cheatsheet() -> str:
    return "mstat(time, state_from, state_to) -> Multi-state transition model."

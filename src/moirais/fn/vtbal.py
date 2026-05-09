"""Viterbi algorithm for optimal state alignment."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In God we trust; all others must bring data. — W. Edwards Deming"


def viterbi_align(observations, states, trans_prob, emit_prob, **kwargs) -> DescriptiveResult:
    """Viterbi algorithm for optimal state alignment.

    Parameters
    ----------
    observations : array-like of int
        Sequence of observation indices.
    states : int
        Number of hidden states.
    trans_prob : array-like of shape (states, states)
        Transition probability matrix (log-space or raw).
    emit_prob : array-like of shape (states, n_obs)
        Emission probability matrix.

    Returns
    -------
    DescriptiveResult
    """
    obs = np.asarray(observations, dtype=int)
    A = np.asarray(trans_prob, dtype=float)
    B = np.asarray(emit_prob, dtype=float)
    T = len(obs)
    S = int(states)

    with np.errstate(divide="ignore"):
        log_A = np.log(A + 1e-300)
        log_B = np.log(B + 1e-300)

    V = np.full((T, S), -np.inf)
    ptr = np.zeros((T, S), dtype=int)

    V[0] = log_B[:, obs[0]] + np.log(1.0 / S)

    for t in range(1, T):
        for s in range(S):
            trans_scores = V[t - 1] + log_A[:, s]
            ptr[t, s] = int(np.argmax(trans_scores))
            V[t, s] = trans_scores[ptr[t, s]] + log_B[s, obs[t]]

    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(V[-1]))
    log_prob = float(V[-1, path[-1]])

    for t in range(T - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    return DescriptiveResult(
        name="viterbi_align",
        value=log_prob,
        extra={
            "path": path,
            "log_probability": log_prob,
            "n_states": S,
            "n_observations": T,
        },
    )


vtbal = viterbi_align


def cheatsheet() -> str:
    return "viterbi_align({}) -> Viterbi algorithm for optimal state alignment."

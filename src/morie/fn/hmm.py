# morie.fn -- function file (hadesllm/morie)
"""Hidden Markov model (forward-backward for discrete obs)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hidden_markov(
    observations: np.ndarray,
    n_states: int = 2,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    seed: int | None = None,
) -> DescriptiveResult:
    """Discrete HMM with Baum-Welch (EM).

    Parameters
    ----------
    observations : (T,) array of integer observation indices
    n_states : int
    max_iter, tol : convergence
    seed : int, optional

    Returns
    -------
    DescriptiveResult
    """
    obs = np.asarray(observations, dtype=int).ravel()
    T = len(obs)
    K = n_states
    M = int(obs.max()) + 1
    rng = np.random.default_rng(seed)

    pi = np.ones(K) / K
    A = rng.dirichlet(np.ones(K), size=K)
    B = rng.dirichlet(np.ones(M), size=K)

    ll_old = -np.inf
    ll_new = -np.inf
    for _ in range(max_iter):
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(pi + 1e-300) + np.log(B[:, obs[0]] + 1e-300)
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = np.logaddexp.reduce(log_alpha[t - 1] + np.log(A[:, j] + 1e-300)) + np.log(
                    B[j, obs[t]] + 1e-300
                )

        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for j in range(K):
                log_beta[t, j] = np.logaddexp.reduce(
                    np.log(A[j, :] + 1e-300) + np.log(B[:, obs[t + 1]] + 1e-300) + log_beta[t + 1]
                )

        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        ll_new = float(np.logaddexp.reduce(log_alpha[-1]))
        if abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

        xi = np.zeros((K, K))
        for t in range(T - 1):
            log_xi_t = np.zeros((K, K))
            for i_s in range(K):
                for j_s in range(K):
                    log_xi_t[i_s, j_s] = (
                        log_alpha[t, i_s]
                        + np.log(A[i_s, j_s] + 1e-300)
                        + np.log(B[j_s, obs[t + 1]] + 1e-300)
                        + log_beta[t + 1, j_s]
                    )
            log_xi_t -= np.logaddexp.reduce(log_xi_t.ravel())
            xi += np.exp(log_xi_t)

        pi = gamma[0]
        A = xi / (xi.sum(axis=1, keepdims=True) + 1e-300)
        for k in range(K):
            for m in range(M):
                B[k, m] = gamma[obs == m, k].sum()
            B[k] /= B[k].sum() + 1e-300

    states = gamma.argmax(axis=1)

    return DescriptiveResult(
        name="hmm",
        value=float(ll_new),
        extra={
            "n_states": K,
            "n_obs_symbols": M,
            "T": T,
            "log_likelihood": float(ll_new),
            "state_counts": [int(np.sum(states == k)) for k in range(K)],
        },
    )


hmm = hidden_markov


def cheatsheet() -> str:
    return "hidden_markov({}) -> Hidden Markov model (forward-backward for discrete obs)."

# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bradley-Terry-Luce model for paired comparisons."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bradley_terry(win_matrix, *, max_iter: int = 100, tol: float = 1e-8) -> DescriptiveResult:
    """Bradley-Terry-Luce model for paired comparison data.

    Estimates ability parameters pi_i such that P(i beats j) = pi_i / (pi_i + pi_j).
    Uses the iterative MM algorithm of Hunter (2004).

    :param win_matrix: n x n matrix where W[i,j] = number of times i beats j.
    :param max_iter: Maximum iterations.
    :param tol: Convergence tolerance.
    :return: DescriptiveResult with estimated strengths and rankings.

    References
    ----------
    Armstrong (2014), Ch 8. Bradley & Terry (1952). Hunter (2004).

    .. epigraph:: "The whole is greater than the sum of its parts. — Aristotle", Star Wars
    """
    W = np.asarray(win_matrix, dtype=float)
    n = W.shape[0]
    if W.shape != (n, n):
        raise ValueError("win_matrix must be square.")

    pi = np.ones(n)
    for _ in range(max_iter):
        pi_old = pi.copy()
        for i in range(n):
            wins_i = W[i].sum()
            denom = 0.0
            for j in range(n):
                if i != j:
                    n_ij = W[i, j] + W[j, i]
                    if n_ij > 0:
                        denom += n_ij / (pi[i] + pi[j])
            pi[i] = wins_i / max(denom, 1e-14)
        pi /= pi.sum()
        if np.max(np.abs(pi - pi_old)) < tol:
            break

    rankings = np.argsort(-pi)
    return DescriptiveResult(
        name="bradley_terry",
        value={"strengths": pi, "rankings": rankings.tolist()},
        extra={"n_items": n, "log_strengths": np.log(pi + 1e-14).tolist()},
    )


btlmd = bradley_terry


def cheatsheet() -> str:
    return "bradley_terry({}) -> Bradley-Terry-Luce paired comparison model."

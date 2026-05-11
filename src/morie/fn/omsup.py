# morie.fn — function file (hadesllm/morie)
"""Minimax optimization. 'Omega Supreme: defend.' -- Omega Supreme"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def minimax_solve(
    payoff_matrix: np.ndarray,
) -> DescriptiveResult:
    """Solve a two-player zero-sum game via minimax / linear programming.

    Finds mixed-strategy Nash equilibrium for the row player using the
    simplex-equivalent LP formulation.

    Parameters
    ----------
    payoff_matrix : ndarray of shape (m, n)
        Payoff matrix for the row player.

    Returns
    -------
    DescriptiveResult
        With ``value`` = game value and ``extra`` containing
        optimal mixed strategies.
    """
    A = np.asarray(payoff_matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("payoff_matrix must be 2-D")
    m, n = A.shape

    shift = 0.0
    if A.min() <= 0:
        shift = abs(A.min()) + 1.0
        A = A + shift

    from scipy.optimize import linprog

    c = np.zeros(m + 1)
    c[-1] = -1

    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -A.T
    A_ub[:, -1] = 1
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        row_strategy = res.x[:m]
        game_value = -res.fun - shift
    else:
        row_strategy = np.ones(m) / m
        maximin = max(A[i].min() for i in range(m))
        game_value = maximin - shift

    c2 = np.zeros(n + 1)
    c2[-1] = 1
    A_ub2 = np.zeros((m, n + 1))
    A_ub2[:, :n] = A
    A_ub2[:, -1] = -1
    b_ub2 = np.zeros(m)
    A_eq2 = np.zeros((1, n + 1))
    A_eq2[0, :n] = 1
    b_eq2 = np.array([1.0])
    bounds2 = [(0, None)] * n + [(None, None)]

    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")

    col_strategy = res2.x[:n] if res2.success else np.ones(n) / n

    return DescriptiveResult(
        name="minimax_solve",
        value=float(game_value),
        extra={"row_strategy": row_strategy, "col_strategy": col_strategy, "m": m, "n": n, "converged": res.success},
    )


omsup = minimax_solve


def cheatsheet() -> str:
    return "minimax_solve({}) -> Minimax optimization. 'Omega Supreme: defend.' -- Omega Supr"

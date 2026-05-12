# morie.fn -- function file (hadesllm/morie)
"""Minimax strategy for zero-sum games."""

import numpy as np
from scipy.optimize import linprog

from ._containers import DescriptiveResult


def minimax(payoff_matrix: np.ndarray) -> DescriptiveResult:
    """
    Compute minimax strategy for a two-player zero-sum game.

    Uses linear programming to find the optimal mixed strategy
    for the row player.

    :param payoff_matrix: (m, n) payoff matrix for the row player.
    :return: DescriptiveResult with optimal strategy and game value.

    References
    ----------
    von Neumann J (1928). Zur Theorie der Gesellschaftsspiele.
    Mathematische Annalen, 100, 295-320.
    """
    A = np.asarray(payoff_matrix, dtype=np.float64)
    m, n = A.shape
    c = np.zeros(m + 1)
    c[-1] = -1
    A_ub = np.column_stack([-A.T, np.ones(n)])
    b_ub = np.zeros(n)
    A_eq = np.ones((1, m + 1))
    A_eq[0, -1] = 0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * m + [(None, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if result.success:
        strategy = result.x[:m]
        value = float(result.x[-1])
    else:
        strategy = np.ones(m) / m
        maxmin = max(min(A[i, :]) for i in range(m))
        minimax_val = min(max(A[:, j]) for j in range(n))
        value = (maxmin + minimax_val) / 2
    return DescriptiveResult(
        name="minimax",
        value=value,
        extra={"strategy": strategy, "game_value": value, "m": m, "n": n},
    )


mnmx = minimax


def cheatsheet() -> str:
    return "minimax({}) -> Minimax strategy for zero-sum games."

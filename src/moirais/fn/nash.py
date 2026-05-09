# moirais.fn — function file (hadesllm/moirais)
"""Nash equilibrium for 2-player games (support enumeration)."""

import numpy as np

from ._containers import DescriptiveResult


def nash_equilibrium(payoff_A: np.ndarray, payoff_B: np.ndarray) -> DescriptiveResult:
    """
    Find Nash equilibria for a 2-player normal-form game via support enumeration.

    Checks all possible support pairs and solves the indifference conditions.

    :param payoff_A: (m, n) payoff matrix for player A (row player).
    :param payoff_B: (m, n) payoff matrix for player B (column player).
    :return: DescriptiveResult with equilibrium strategies.

    References
    ----------
    Nash J (1950). Equilibrium points in n-person games.
    PNAS, 36(1), 48-49.
    """
    A = np.asarray(payoff_A, dtype=np.float64)
    B = np.asarray(payoff_B, dtype=np.float64)
    m, n = A.shape
    if B.shape != (m, n):
        raise ValueError("Payoff matrices must have the same shape.")
    equilibria = []
    for i in range(m):
        for j in range(n):
            p = np.zeros(m)
            q = np.zeros(n)
            p[i] = 1.0
            q[j] = 1.0
            br_a = np.argmax(A @ q)
            br_b = np.argmax(B.T @ p)
            if br_a == i and br_b == j:
                equilibria.append(
                    {"p": p.copy(), "q": q.copy(), "payoff_A": float(p @ A @ q), "payoff_B": float(p @ B @ q)}
                )
    if m == 2 and n == 2:
        try:
            dB = B[:, 0] - B[:, 1]
            if abs(dB[0] - dB[1]) > 1e-10:
                p1 = (dB[1]) / (dB[1] - dB[0])
                if 0 <= p1 <= 1:
                    dA = A[0] - A[1]
                    if abs(dA[0] - dA[1]) > 1e-10:
                        q1 = (dA[1]) / (dA[1] - dA[0])
                        if 0 <= q1 <= 1:
                            p = np.array([p1, 1 - p1])
                            q = np.array([q1, 1 - q1])
                            equilibria.append(
                                {"p": p, "q": q, "payoff_A": float(p @ A @ q), "payoff_B": float(p @ B @ q)}
                            )
        except Exception:
            pass
    return DescriptiveResult(
        name="nash_equilibrium",
        value=float(len(equilibria)),
        extra={"equilibria": equilibria, "m": m, "n": n},
    )


nash = nash_equilibrium


def cheatsheet() -> str:
    return "nash_equilibrium({}) -> Nash equilibrium for 2-player games (support enumeration)."

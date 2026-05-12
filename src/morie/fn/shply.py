"""Shapley values for cooperative games."""

from itertools import permutations

import numpy as np

from ._containers import DescriptiveResult


def shapley_value(characteristic_fn, n_players: int) -> DescriptiveResult:
    r"""
    Compute exact Shapley values for a cooperative game.

    .. math::

        \\phi_i = \\sum_{S \\subseteq N \\setminus \\{i\\}}
        \\frac{|S|!(n-|S|-1)!}{n!} [v(S \\cup \\{i\\}) - v(S)]

    :param characteristic_fn: Callable taking a frozenset of players and
        returning the coalition value.
    :param n_players: Number of players.
    :return: DescriptiveResult with Shapley values.

    References
    ----------
    Shapley LS (1953). A value for n-person games. In: Kuhn HW,
    Tucker AW (eds). Contributions to the Theory of Games II.
    Princeton University Press, 307-317.
    """
    if n_players > 10:
        raise ValueError("Exact Shapley values infeasible for n > 10.")
    players = list(range(n_players))
    phi = np.zeros(n_players)
    n_perms = 0
    for perm in permutations(players):
        coalition = set()
        for player in perm:
            v_with = characteristic_fn(frozenset(coalition | {player}))
            v_without = characteristic_fn(frozenset(coalition))
            phi[player] += v_with - v_without
            coalition.add(player)
        n_perms += 1
    phi /= n_perms
    return DescriptiveResult(
        name="shapley_value",
        value=float(phi.sum()),
        extra={"values": phi, "n_players": n_players, "efficiency": float(characteristic_fn(frozenset(players)))},
    )


shply = shapley_value


def cheatsheet() -> str:
    return "shapley_value({}) -> Shapley values for cooperative games."

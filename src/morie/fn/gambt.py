# morie.fn — function file (hadesllm/morie)
"""Combinatorial probability (card counting). 'The name is Gambit. Remember it.' -- Gambit"""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def card_probability(
    n_total: int = 52,
    n_target: int = 4,
    draw: int = 5,
    *,
    at_least: int = 1,
) -> DescriptiveResult:
    r"""Compute the probability of drawing at least *at_least* target cards
    in a *draw*-card hand from a deck via the hypergeometric distribution.

    :math:`P(X \\ge k) = \\sum_{i=k}^{\\min(draw, n_{target})}`
    :math:`\\frac{\\binom{n_t}{i} \\binom{n - n_t}{draw - i}}{\\binom{n}{draw}}`

    Parameters
    ----------
    n_total : int
        Total cards in the deck.
    n_target : int
        Number of target cards (e.g., 4 aces).
    draw : int
        Hand size.
    at_least : int
        Minimum number of target cards desired.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``probability``, ``pmf`` (dict: k -> P(X=k)),
        ``expected``, ``variance``.
    """
    if n_total < 1 or n_target < 0 or draw < 0 or at_least < 0:
        raise ValueError("All counts must be non-negative, n_total >= 1")
    if n_target > n_total or draw > n_total:
        raise ValueError("n_target and draw must be <= n_total")

    def comb(a, b):
        if b < 0 or b > a:
            return 0
        return math.comb(a, b)

    total_ways = comb(n_total, draw)
    pmf = {}
    for k in range(max(0, draw - (n_total - n_target)), min(draw, n_target) + 1):
        pmf[k] = comb(n_target, k) * comb(n_total - n_target, draw - k) / total_ways

    prob = sum(v for k, v in pmf.items() if k >= at_least)
    expected = draw * n_target / n_total
    variance = (
        (draw * n_target * (n_total - n_target) * (n_total - draw) / (n_total**2 * (n_total - 1)))
        if n_total > 1
        else 0.0
    )

    return DescriptiveResult(
        name="card_probability",
        value={
            "probability": float(prob),
            "pmf": pmf,
            "expected": float(expected),
            "variance": float(variance),
        },
        extra={"n_total": n_total, "n_target": n_target, "draw": draw, "at_least": at_least},
    )


gambt = card_probability


def cheatsheet() -> str:
    return "card_probability({}) -> Combinatorial probability (card counting). 'The name is Gamb"

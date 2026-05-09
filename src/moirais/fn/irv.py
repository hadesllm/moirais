# moirais.fn — function file (hadesllm/moirais)
"""Instant runoff voting (ranked choice)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def instant_runoff(rankings) -> DescriptiveResult:
    """Instant runoff voting (IRV / ranked-choice voting).

    Iteratively eliminates the candidate with fewest first-place votes
    until one candidate has a majority.

    :param rankings: (n_voters x n_candidates) matrix of rankings (1=best).
    :return: DescriptiveResult with winner and elimination order.

    References
    ----------
    Armstrong (2014), Ch 1.

    .. epigraph:: 'Confine yourself to the present. — Marcus Aurelius'
    """
    R = np.asarray(rankings, dtype=float)
    if R.ndim != 2:
        raise ValueError("rankings must be 2D (voters x candidates).")
    n_voters, n_cand = R.shape
    active = set(range(n_cand))
    elimination_order = []
    rounds = []

    for _ in range(n_cand - 1):
        first_place = np.full(n_voters, -1, dtype=int)
        for i in range(n_voters):
            best_rank = n_cand + 1
            for c in active:
                if R[i, c] < best_rank:
                    best_rank = R[i, c]
                    first_place[i] = c

        counts = {c: int(np.sum(first_place == c)) for c in active}
        rounds.append(dict(counts))

        max_count = max(counts.values())
        if max_count > n_voters / 2:
            winner = max(counts, key=counts.get)
            return DescriptiveResult(
                name="instant_runoff",
                value=winner,
                extra={"elimination_order": elimination_order, "rounds": rounds},
            )

        min_count = min(counts.values())
        eliminated = min(c for c in active if counts[c] == min_count)
        elimination_order.append(eliminated)
        active.remove(eliminated)

    winner = active.pop() if active else -1
    return DescriptiveResult(
        name="instant_runoff",
        value=winner,
        extra={"elimination_order": elimination_order, "rounds": rounds},
    )


irv = instant_runoff


def cheatsheet() -> str:
    return "instant_runoff({}) -> Instant runoff (ranked choice) voting."

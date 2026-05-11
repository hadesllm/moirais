# morie.fn — function file (hadesllm/morie)
"""Read roll-call vote matrix."""

from __future__ import annotations

from ._containers import DescriptiveResult


def read_roll_call(votes, yea=1, nay=0) -> DescriptiveResult:
    """Format a raw vote matrix with yea/nay coding and missing handling.

    .. epigraph:: "I am the danger." -- Walter White, Breaking Bad
    """
    import numpy as np

    votes = np.asarray(votes, dtype=float)
    yea_count = int(np.nansum(votes == yea))
    nay_count = int(np.nansum(votes == nay))
    missing = int(np.sum(np.isnan(votes)))
    n_leg, n_votes = votes.shape if votes.ndim == 2 else (1, votes.shape[0])
    return DescriptiveResult(
        name="read_roll_call",
        value=float(yea_count),
        extra={
            "yea_count": yea_count,
            "nay_count": nay_count,
            "missing": missing,
            "n_legislators": n_leg,
            "n_votes": n_votes,
            "matrix": votes,
        },
    )


rcred = read_roll_call


def cheatsheet() -> str:
    return "read_roll_call({}) -> Read roll-call vote matrix."

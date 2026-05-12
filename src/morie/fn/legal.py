# morie.fn -- function file (hadesllm/morie)
"""Legal BAC limit threshold check."""

from __future__ import annotations


def is_over_legal_limit(ebac: float, limit: float = 0.08) -> int:
    """Determine whether an eBAC exceeds the legal driving limit.

    Parameters
    ----------
    ebac : float
        Calculated eBAC value.
    limit : float
        Legal BAC limit (default 0.08).

    Returns
    -------
    int
        1 if ebac >= limit, 0 otherwise.
    """
    return 1 if ebac >= limit else 0


legal = is_over_legal_limit


def cheatsheet() -> str:
    return "is_over_legal_limit({}) -> Legal BAC limit threshold check."

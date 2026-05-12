# morie.fn -- function file (hadesllm/morie)
"""Civil liberties metric (Charter challenges, stays)."""

from __future__ import annotations

from morie.fn._containers import CrimeResult


def court_civil_liberties(
    n_challenges: int,
    n_granted: int,
) -> CrimeResult:
    """Compute Charter challenge success rate.

    Parameters
    ----------
    n_challenges : int
        Number of Charter challenges filed.
    n_granted : int
        Number of challenges granted (evidence excluded, stay of proceedings).

    Returns
    -------
    CrimeResult
    """
    if n_challenges <= 0:
        raise ValueError("n_challenges must be positive")
    if n_granted > n_challenges:
        raise ValueError("n_granted cannot exceed n_challenges")
    rate = n_granted / n_challenges
    return CrimeResult(
        name="charter_challenge_rate",
        rate=rate,
        n=n_granted,
        population=n_challenges,
        extra={"n_denied": n_challenges - n_granted},
    )


crtcv = court_civil_liberties


def cheatsheet() -> str:
    return "court_civil_liberties({}) -> Civil liberties metric (Charter challenges, stays)."

# moirais.fn — function file (hadesllm/moirais)
"""Bail grant rate analysis."""

from __future__ import annotations

from moirais.fn._containers import CrimeResult


def court_bail_rate(
    n_granted: int,
    n_hearings: int,
) -> CrimeResult:
    """Compute bail grant rate.

    Parameters
    ----------
    n_granted : int
    n_hearings : int

    Returns
    -------
    CrimeResult
    """
    if n_hearings <= 0:
        raise ValueError("n_hearings must be positive")
    rate = n_granted / n_hearings
    return CrimeResult(
        name="bail_grant_rate",
        rate=rate,
        n=n_granted,
        population=n_hearings,
        extra={"n_denied": n_hearings - n_granted, "denial_rate": 1 - rate},
    )


crtbl = court_bail_rate


def cheatsheet() -> str:
    return "court_bail_rate({}) -> Bail grant rate analysis."

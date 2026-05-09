"""Civilian complaint rate."""

from __future__ import annotations

from moirais.fn._containers import CrimeResult


def tps_complaint_rate(
    n_complaints: int,
    n_officers: int,
    *,
    per: int = 1000,
) -> CrimeResult:
    """Compute civilian complaint rate per N officers.

    Parameters
    ----------
    n_complaints : int
        Total civilian complaints filed.
    n_officers : int
        Total sworn officers.
    per : int
        Rate denominator (default 1000).

    Returns
    -------
    CrimeResult
    """
    if n_officers <= 0:
        raise ValueError("n_officers must be positive")
    rate = n_complaints / n_officers * per
    return CrimeResult(
        name="complaint_rate",
        rate=rate,
        n=n_complaints,
        population=n_officers,
        extra={"per": per},
    )


tpscmp = tps_complaint_rate


def cheatsheet() -> str:
    return "tps_complaint_rate({}) -> Civilian complaint rate."

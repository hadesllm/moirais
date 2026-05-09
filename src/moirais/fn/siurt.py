"""SIU case rate per 1000 officers."""

from __future__ import annotations

from moirais.fn._containers import CrimeResult


def siu_case_rate(
    n_cases: int,
    n_officers: int,
    *,
    per: int = 1000,
) -> CrimeResult:
    """Compute SIU case rate per N officers.

    Parameters
    ----------
    n_cases : int
        Number of SIU investigations.
    n_officers : int
        Total sworn officers in jurisdiction.
    per : int
        Rate denominator (default 1000).

    Returns
    -------
    CrimeResult
    """
    if n_officers <= 0:
        raise ValueError("n_officers must be positive")
    rate = n_cases / n_officers * per
    return CrimeResult(name="siu_case_rate", rate=rate, n=n_cases, population=n_officers, extra={"per": per})


siurt = siu_case_rate


def cheatsheet() -> str:
    return "siu_case_rate({}) -> SIU case rate per 1000 officers."

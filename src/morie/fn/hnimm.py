# morie.fn -- function file (hadesllm/morie)
"""Herd immunity threshold computation."""

from __future__ import annotations

from typing import Any


def herd_immunity_threshold(
    R0: float,
    *,
    vaccine_efficacy: float = 1.0,
) -> dict[str, Any]:
    """Compute the herd immunity threshold (HIT).

    .. math::

        HIT = 1 - \\frac{1}{R_0}

    With imperfect vaccine efficacy (VE < 1), the critical vaccination
    coverage is:

    .. math::

        V_c = \\frac{1 - 1/R_0}{VE}

    Parameters
    ----------
    R0 : float
        Basic reproduction number (must be > 0).
    vaccine_efficacy : float, default 1.0
        Vaccine efficacy (0 to 1).

    Returns
    -------
    dict
        Keys: 'hit' (natural herd immunity threshold),
              'critical_vaccination_coverage',
              'R0', 'vaccine_efficacy'.

    References
    ----------
    Fine, P., Eames, K., & Heymann, D. L. (2011). "Herd immunity":
    a rough guide. Clinical Infectious Diseases, 52(7), 911-916.
    """
    if R0 <= 0:
        raise ValueError("R0 must be positive.")
    if not (0 < vaccine_efficacy <= 1.0):
        raise ValueError("vaccine_efficacy must be in (0, 1].")

    hit = 1.0 - 1.0 / R0
    vc = hit / vaccine_efficacy

    return {
        "hit": float(hit),
        "critical_vaccination_coverage": float(min(vc, 1.0)),
        "R0": float(R0),
        "vaccine_efficacy": float(vaccine_efficacy),
    }


hnimm = herd_immunity_threshold


def cheatsheet() -> str:
    return "herd_immunity_threshold({}) -> HIT = 1 - 1/R0."

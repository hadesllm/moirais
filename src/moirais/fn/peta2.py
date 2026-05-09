# moirais.fn — function file (hadesllm/moirais)
"""Partial eta-squared effect size."""

from ._containers import ESRes


def partial_eta_squared(
    ss_effect: float,
    ss_error: float,
) -> ESRes:
    """Partial eta-squared.

    eta^2_p = SS_effect / (SS_effect + SS_error)

    Parameters
    ----------
    ss_effect, ss_error : float

    Returns
    -------
    ESRes
    """
    denom = ss_effect + ss_error
    pe2 = ss_effect / denom if denom > 0 else 0.0
    return ESRes(
        measure="Partial eta-squared",
        estimate=float(pe2),
    )


peta2 = partial_eta_squared


def cheatsheet() -> str:
    return "partial_eta_squared({}) -> Partial eta-squared effect size."

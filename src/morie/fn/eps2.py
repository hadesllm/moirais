# morie.fn — function file (hadesllm/morie)
"""Epsilon-squared effect size (Kelley, 1935)."""

from ._containers import ESRes


def epsilon_squared(
    ss_effect: float,
    ss_total: float,
    df_effect: int,
    ms_error: float,
) -> ESRes:
    """Epsilon-squared (Kelley, 1935).

    eps^2 = (SS_effect - df_effect * MS_error) / SS_total

    Parameters
    ----------
    ss_effect, ss_total : float
    df_effect : int
    ms_error : float

    Returns
    -------
    ESRes
    """
    num = ss_effect - df_effect * ms_error
    eps2 = max(num / ss_total, 0.0) if ss_total > 0 else 0.0
    return ESRes(
        measure="Epsilon-squared",
        estimate=float(eps2),
    )


eps2 = epsilon_squared


def cheatsheet() -> str:
    return "epsilon_squared({}) -> Epsilon-squared effect size (Kelley, 1935)."

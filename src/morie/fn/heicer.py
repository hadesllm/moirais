# morie.fn -- function file (hadesllm/morie)
"""Incremental cost-effectiveness ratio (ICER)."""

from ._containers import ESRes


def incremental_cer(
    cost_new: float,
    cost_old: float,
    effect_new: float,
    effect_old: float,
) -> ESRes:
    """Compute ICER = (C_new - C_old) / (E_new - E_old).

    Parameters
    ----------
    cost_new, cost_old : float
    effect_new, effect_old : float

    Returns
    -------
    ESRes
    """
    delta_c = cost_new - cost_old
    delta_e = effect_new - effect_old

    if delta_e == 0:
        import numpy as np

        icer = np.inf if delta_c > 0 else -np.inf if delta_c < 0 else 0.0
    else:
        icer = delta_c / delta_e

    if delta_c < 0 and delta_e > 0:
        quadrant = "SE_dominant"
    elif delta_c > 0 and delta_e < 0:
        quadrant = "NW_dominated"
    elif delta_c > 0 and delta_e > 0:
        quadrant = "NE_tradeoff"
    else:
        quadrant = "SW_tradeoff"

    return ESRes(
        measure="ICER",
        estimate=float(icer),
        extra={"delta_cost": float(delta_c), "delta_effect": float(delta_e), "quadrant": quadrant},
    )


heicer = incremental_cer


def cheatsheet() -> str:
    return "incremental_cer({}) -> Incremental cost-effectiveness ratio (ICER)."

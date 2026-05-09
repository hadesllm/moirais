# moirais.fn — function file (hadesllm/moirais)
"""Incremental cost-effectiveness ratio (ICER)."""

from __future__ import annotations

from typing import Any

import numpy as np


def incremental_cost_effectiveness_ratio(
    cost_new: float,
    cost_ref: float,
    effect_new: float,
    effect_ref: float,
    *,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute the incremental cost-effectiveness ratio (ICER).

    .. math::

        ICER = \\frac{C_{\\text{new}} - C_{\\text{ref}}}{E_{\\text{new}} - E_{\\text{ref}}}

    Parameters
    ----------
    cost_new : float
        Total cost of the new intervention.
    cost_ref : float
        Total cost of the reference/comparator.
    effect_new : float
        Health effect (e.g. QALYs) of the new intervention.
    effect_ref : float
        Health effect of the reference.
    n_bootstrap : int, default 0
        If > 0, number of bootstrap samples for CE plane scatter
        (requires cost/effect to be mean estimates).
    seed : int, default 42
        Random seed for bootstrap.

    Returns
    -------
    dict
        Keys: 'icer', 'delta_cost', 'delta_effect', 'quadrant'.
        If n_bootstrap > 0: 'bootstrap_icers'.

    References
    ----------
    Drummond, M. F. et al. (2015). *Methods for the Economic Evaluation
    of Health Care Programmes*, 4th ed. Oxford University Press, Ch. 4.
    """
    delta_c = cost_new - cost_ref
    delta_e = effect_new - effect_ref

    if abs(delta_e) < 1e-15:
        icer = np.inf if delta_c > 0 else (-np.inf if delta_c < 0 else 0.0)
    else:
        icer = delta_c / delta_e

    if delta_c >= 0 and delta_e >= 0:
        quadrant = "NE (more costly, more effective)"
    elif delta_c < 0 and delta_e >= 0:
        quadrant = "SE (less costly, more effective — dominant)"
    elif delta_c >= 0 and delta_e < 0:
        quadrant = "NW (more costly, less effective — dominated)"
    else:
        quadrant = "SW (less costly, less effective)"

    result: dict[str, Any] = {
        "icer": float(icer),
        "delta_cost": float(delta_c),
        "delta_effect": float(delta_e),
        "quadrant": quadrant,
    }

    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        se_c = abs(delta_c) * 0.1 if abs(delta_c) > 0 else 1.0
        se_e = abs(delta_e) * 0.1 if abs(delta_e) > 0 else 0.01
        boot_dc = rng.normal(delta_c, se_c, n_bootstrap)
        boot_de = rng.normal(delta_e, se_e, n_bootstrap)
        valid = np.abs(boot_de) > 1e-15
        boot_icer = np.where(valid, boot_dc / boot_de, np.nan)
        result["bootstrap_icers"] = boot_icer

    return result


icerr = incremental_cost_effectiveness_ratio


def cheatsheet() -> str:
    return "incremental_cost_effectiveness_ratio({}) -> ICER."

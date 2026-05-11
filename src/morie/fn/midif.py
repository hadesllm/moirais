# morie.fn — function file (hadesllm/morie)
"""Delta-fit indices between measurement invariance levels."""

from __future__ import annotations

from morie.fn._mapq_const import MI_DELTA


def mi_delta_fit(
    fit1: dict,
    fit2: dict,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Compute delta-fit indices between two invariance levels.

    Conventional criteria (Chen, 2007): |delta_CFI| <= 0.01,
    |delta_RMSEA| <= 0.015.

    Parameters
    ----------
    fit1 : dict
        Fit indices from the less constrained model (requires 'cfi', 'rmsea').
    fit2 : dict
        Fit indices from the more constrained model.
    thresholds : dict, optional
        Override threshold dict. Default: MI_DELTA from _mapq_const.

    Returns
    -------
    dict
        Keys: delta_cfi, delta_rmsea, delta_srmr, passed.

    References
    ----------
    Chen, F.F. (2007). Sensitivity of goodness of fit indexes to lack of
        measurement invariance. SEM, 14(3), 464-504.
    """
    thresh = thresholds if thresholds is not None else MI_DELTA

    # Extract fit dicts (handle nested 'fit' key)
    f1 = fit1.get("fit", fit1)
    f2 = fit2.get("fit", fit2)

    delta_cfi = f1.get("cfi", 1.0) - f2.get("cfi", 1.0)
    delta_rmsea = f2.get("rmsea", 0.0) - f1.get("rmsea", 0.0)
    delta_srmr = f2.get("srmr", 0.0) - f1.get("srmr", 0.0)

    passed = abs(delta_cfi) <= thresh.get("delta_cfi", 0.01) and abs(delta_rmsea) <= thresh.get("delta_rmsea", 0.015)

    return {
        "delta_cfi": float(delta_cfi),
        "delta_rmsea": float(delta_rmsea),
        "delta_srmr": float(delta_srmr),
        "passed": bool(passed),
    }


def cheatsheet() -> str:
    return "mi_delta_fit({}) -> Delta-fit indices between measurement invariance levels."

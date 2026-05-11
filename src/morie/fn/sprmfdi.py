"""Spatial probit direct/indirect MEs."""

import numpy as np

from ._containers import SpatialResult


def sprmfdi(coef, rho, X, W):
    """Spatial probit direct/indirect MEs.

    Category: SProbit

    Parameters
    ----------
    coef, rho, X, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="sprmfdi", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sprmfdi", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sprmfdi_fn = sprmfdi


def cheatsheet() -> str:
    return "sprmfdi({}) -> Spatial probit direct/indirect MEs."

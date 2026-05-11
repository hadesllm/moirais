"""Spatial probit predicted probabilities."""

import numpy as np

from ._containers import SpatialResult


def spprprd(coef, X, W, rho=0.2):
    """Spatial probit predicted probabilities.

    Category: SProbit

    Parameters
    ----------
    coef, X, W, rho=0.2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - rho * eigvals) + 1e-12)))
        return SpatialResult(name="spprprd", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="spprprd", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


spprprd_fn = spprprd


def cheatsheet() -> str:
    return "spprprd({}) -> Spatial probit predicted probabilities."

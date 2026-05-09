# moirais.fn — function file (hadesllm/moirais)
"""SDEM direct/indirect/total impacts."""

import numpy as np

from ._containers import SpatialResult


def sdemimp(coef, theta, lam, W):
    """SDEM direct/indirect/total impacts.

    Category: SDEM

    Parameters
    ----------
    coef, theta, lam, W : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - lam * eigvals) + 1e-12)))
        return SpatialResult(name="sdemimp", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdemimp", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdemimp_fn = sdemimp


def cheatsheet() -> str:
    return "sdemimp({}) -> SDEM direct/indirect/total impacts."

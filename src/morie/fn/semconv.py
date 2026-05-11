# morie.fn — function file (hadesllm/morie)
"""SEM convergence check (lambda feasibility)."""

import numpy as np

from ._containers import SpatialResult


def semconv(W, lam):
    """SEM convergence check (lambda feasibility).

    Category: SEM

    Parameters
    ----------
    W, lam : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        eigvals = np.linalg.eigvalsh(W)
        result = float(np.sum(np.log(np.abs(1 - lam * eigvals) + 1e-12)))
        return SpatialResult(name="semconv", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="semconv", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


semconv_fn = semconv


def cheatsheet() -> str:
    return "semconv({}) -> SEM convergence check (lambda feasibility)."

# morie.fn — function file (hadesllm/morie)
"""SDEM Jacobian term."""

import numpy as np

from ._containers import SpatialResult


def sdemjac(W, lam):
    """SDEM Jacobian term.

    Category: SDEM

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
        return SpatialResult(name="sdemjac", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(
            name="sdemjac", statistic=float("nan"), p_value=None, extra={"error": "computation failed"}
        )


sdemjac_fn = sdemjac


def cheatsheet() -> str:
    return "sdemjac({}) -> SDEM Jacobian term."

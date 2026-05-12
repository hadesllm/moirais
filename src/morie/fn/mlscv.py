# morie.fn -- function file (hadesllm/morie)
"""MLSMU6 convergence check. 'Jagan Eye.' -- Hiei, Yu Yu Hakusho"""

from __future__ import annotations

from ._containers import DescriptiveResult


def mlsmu6_convergence_check(stress_old, stress_new, tol=1e-6):
    """Check convergence of MLSMU6 unfolding.

    Parameters
    ----------
    stress_old : float
        Previous iteration stress.
    stress_new : float
        Current iteration stress.
    tol : float
        Convergence tolerance.

    Returns
    -------
    DescriptiveResult
        value = bool (converged).
    """
    delta = abs(stress_old - stress_new)
    converged = delta < tol
    return DescriptiveResult(
        name="mlsmu6_convergence_check",
        value=converged,
        extra={"delta": delta, "tol": tol},
    )


mlscv = mlsmu6_convergence_check


def cheatsheet() -> str:
    return "mlsmu6_convergence_check({}) -> MLSMU6 convergence check. 'Jagan Eye.' -- Hiei, Yu Yu Hakush"

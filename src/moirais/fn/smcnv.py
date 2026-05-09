"""SMACOF convergence check. 'Serious Series.' -- Saitama, One Punch Man"""

from __future__ import annotations

from ._containers import DescriptiveResult


def smacof_convergence(stress_old, stress_new, tol=1e-6):
    """Check if SMACOF has converged.

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
        value = bool (converged), extra has delta.
    """
    delta = abs(stress_old - stress_new)
    converged = delta < tol
    return DescriptiveResult(
        name="smacof_convergence",
        value=converged,
        extra={"delta": delta, "tol": tol, "stress_old": stress_old, "stress_new": stress_new},
    )


smcnv = smacof_convergence


def cheatsheet() -> str:
    return "smacof_convergence({}) -> SMACOF convergence check. 'Serious Series.' -- Saitama, One "

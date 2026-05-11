# morie.fn — function file (hadesllm/morie)
"""EM convergence check."""

from __future__ import annotations

from ._containers import DescriptiveResult


def em_convergence_check(ll_old, ll_new, tol=1e-6) -> DescriptiveResult:
    """Check if EM has converged based on log-likelihood change.

    .. epigraph:: "When you play the game of thrones..." -- Cersei, Game of Thrones
    """
    diff = abs(ll_new - ll_old)
    converged = diff < tol
    return DescriptiveResult(
        name="em_convergence_check",
        value=float(diff),
        extra={
            "converged": converged,
            "diff": float(diff),
            "tol": tol,
            "ll_old": float(ll_old),
            "ll_new": float(ll_new),
        },
    )


emcnv = em_convergence_check


def cheatsheet() -> str:
    return "em_convergence_check({}) -> EM convergence check."

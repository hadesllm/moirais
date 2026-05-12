# morie.fn -- function file (hadesllm/morie)
"""Court backlog analysis."""

from __future__ import annotations

from morie.fn._containers import DescriptiveResult


def court_backlog(
    pending: int,
    resolved: int,
    new_cases: int,
) -> DescriptiveResult:
    """Analyse court case backlog.

    Parameters
    ----------
    pending : int
        Currently pending cases.
    resolved : int
        Cases resolved in period.
    new_cases : int
        New cases filed in period.

    Returns
    -------
    DescriptiveResult
    """
    if resolved <= 0:
        raise ValueError("resolved must be positive")
    clearance_rate = resolved / new_cases if new_cases > 0 else float("inf")
    backlog_ratio = pending / resolved
    months_to_clear = pending / (resolved / 12) if resolved > 0 else float("inf")
    return DescriptiveResult(
        name="court_backlog",
        value=float(pending),
        extra={
            "pending": pending,
            "resolved": resolved,
            "new_cases": new_cases,
            "clearance_rate": clearance_rate,
            "backlog_ratio": backlog_ratio,
            "months_to_clear": months_to_clear,
        },
    )


crtbk = court_backlog


def cheatsheet() -> str:
    return "court_backlog({}) -> Court backlog analysis."

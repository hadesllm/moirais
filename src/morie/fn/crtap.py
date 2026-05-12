# morie.fn -- function file (hadesllm/morie)
"""Appeal rate and success."""

from __future__ import annotations

from morie.fn._containers import DescriptiveResult


def court_appeal_rate(
    n_appeals: int,
    n_convictions: int,
    n_overturned: int,
) -> DescriptiveResult:
    """Compute appeal rate and overturn rate.

    Parameters
    ----------
    n_appeals : int
        Number of appeals filed.
    n_convictions : int
        Total convictions in period.
    n_overturned : int
        Appeals resulting in overturned conviction.

    Returns
    -------
    DescriptiveResult
    """
    if n_convictions <= 0:
        raise ValueError("n_convictions must be positive")
    appeal_rate = n_appeals / n_convictions
    overturn_rate = n_overturned / n_appeals if n_appeals > 0 else 0.0
    return DescriptiveResult(
        name="court_appeal",
        value=appeal_rate,
        extra={
            "appeal_rate": appeal_rate,
            "overturn_rate": overturn_rate,
            "n_appeals": n_appeals,
            "n_convictions": n_convictions,
            "n_overturned": n_overturned,
        },
    )


crtap = court_appeal_rate


def cheatsheet() -> str:
    return "court_appeal_rate({}) -> Appeal rate and success."

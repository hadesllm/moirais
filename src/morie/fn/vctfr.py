"""Fear of crime index."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def victim_fear(
    survey_responses: np.ndarray | list[float],
) -> DescriptiveResult:
    """Compute fear-of-crime index from Likert survey responses.

    Parameters
    ----------
    survey_responses : array-like
        Likert scale responses (e.g. 1-5).

    Returns
    -------
    DescriptiveResult
    """
    s = np.asarray(survey_responses, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        raise ValueError("No valid responses")
    return DescriptiveResult(
        name="fear_of_crime_index",
        value=float(np.mean(s)),
        extra={
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
            "n": len(s),
            "pct_high_fear": float(np.mean(s >= 4)) if np.max(s) >= 4 else 0.0,
        },
    )


vctfr = victim_fear


def cheatsheet() -> str:
    return "victim_fear({}) -> Fear of crime index."

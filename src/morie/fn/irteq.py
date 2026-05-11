# morie.fn — function file (hadesllm/morie)
"""IRT true-score equating."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def irt_equating(
    params_a: dict,
    params_b: dict,
    *,
    theta_grid: np.ndarray | None = None,
) -> DescriptiveResult:
    """IRT true-score equating between two test forms.

    For each theta, computes expected true score on both forms
    and builds a concordance table.

    Parameters
    ----------
    params_a : dict
        {item: {"a": float, "b": float}} for form A.
    params_b : dict
        Same for form B.
    theta_grid : ndarray, optional
        Default linspace(-4, 4, 81).

    Returns
    -------
    DescriptiveResult
        value=dict with true_score_a, true_score_b, theta arrays.

    References
    ----------
    Lord, F. M. (1980). Applications of Item Response Theory
    to Practical Testing Problems. Lawrence Erlbaum.
    """
    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 81)

    def expected_score(params, theta):
        score = np.zeros_like(theta)
        for p in params.values():
            a = p.get("a", 1.0)
            b = p.get("b", 0.0)
            logit = a * (theta - b)
            P = 1.0 / (1.0 + np.exp(-np.clip(logit, -700, 700)))
            score += P
        return score

    ts_a = expected_score(params_a, theta_grid)
    ts_b = expected_score(params_b, theta_grid)

    return DescriptiveResult(
        name="IRT equating",
        value={
            "theta": theta_grid.tolist(),
            "true_score_a": ts_a.tolist(),
            "true_score_b": ts_b.tolist(),
        },
        extra={"k_a": len(params_a), "k_b": len(params_b)},
    )


equating = irt_equating


def cheatsheet() -> str:
    return "irt_equating({}) -> IRT true-score equating."

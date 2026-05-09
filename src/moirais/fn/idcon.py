# moirais.fn — function file (hadesllm/moirais)
"""Ideological constraint index from issue positions."""

from __future__ import annotations

from ._containers import DescriptiveResult


def ideological_constraint(issue_positions) -> DescriptiveResult:
    """Compute ideological constraint as mean pairwise correlation of issues.

    :param issue_positions: Respondent x issue matrix.
    :return: DescriptiveResult with constraint index.

    .. epigraph:: "I am the bone of my sword." -- Archer, Fate/stay night
    """
    import numpy as np

    X = np.asarray(issue_positions, dtype=float)
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairwise = corr[mask]
    constraint = float(np.mean(np.abs(pairwise)))
    return DescriptiveResult(
        name="ideological_constraint",
        value=constraint,
        extra={"n_issues": n, "mean_abs_corr": constraint, "corr_matrix": corr.tolist()},
    )


idcon = ideological_constraint


def cheatsheet() -> str:
    return "ideological_constraint({}) -> Ideological constraint index from issue positions."

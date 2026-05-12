# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Blackbox R-squared per issue scale."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bb_r2_per_issue(Z, fitted) -> DescriptiveResult:
    """Compute R-squared for each issue column in Blackbox.

    :param Z: Original respondent x issue data.
    :param fitted: Fitted values from Blackbox reconstruction.
    :return: DescriptiveResult with per-issue R-squared.

    .. epigraph:: "Yare yare daze." -- Jotaro Kujo, JoJo's Bizarre Adventure
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    n_issues = Z.shape[1]
    r2s = []
    for j in range(n_issues):
        y = Z[:, j]
        yhat = fitted[:, j]
        mask = ~np.isnan(y)
        y_m, yhat_m = y[mask], yhat[mask]
        ss_tot = np.sum((y_m - y_m.mean()) ** 2)
        ss_res = np.sum((y_m - yhat_m) ** 2)
        r2s.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    return DescriptiveResult(
        name="bb_r2_per_issue",
        value=float(np.mean(r2s)),
        extra={"r2_per_issue": r2s, "n_issues": n_issues},
    )


bbr2 = bb_r2_per_issue


def cheatsheet() -> str:
    return "bb_r2_per_issue({}) -> Blackbox R-squared per issue scale."

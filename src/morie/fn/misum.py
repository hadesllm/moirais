# morie.fn — function file (hadesllm/morie)
"""Summary table of invariance levels with pass/fail."""

from __future__ import annotations

import pandas as pd


def mi_summary(
    fit_list: list[dict],
    *,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Summary table of measurement invariance testing results.

    Parameters
    ----------
    fit_list : list of dict
        List of invariance fit dicts (from mi_by_gender, mi_by_age, etc.).
        Each dict should have keys: level, fit, delta_fit, passed.
    labels : list of str, optional
        Custom level labels. Default: extracted from each dict's 'level' key.

    Returns
    -------
    DataFrame
        Columns: level, chi2, df, cfi, tli, rmsea, srmr, delta_cfi,
        delta_rmsea, passed, highest_invariance.

    References
    ----------
    Putnick, D.L. & Bornstein, M.H. (2016). Measurement invariance
        conventions and reporting. Developmental Review, 41, 71-90.
    """
    rows = []
    highest = "none"

    for i, fit in enumerate(fit_list):
        level = fit.get("level", labels[i] if labels and i < len(labels) else f"level_{i}")
        f = fit.get("fit", {})
        delta = fit.get("delta_fit", {})
        passed = fit.get("passed", False)

        if passed:
            highest = level

        rows.append(
            {
                "level": level,
                "chi2": f.get("chi2", float("nan")),
                "df": f.get("df", 0),
                "cfi": f.get("cfi", float("nan")),
                "tli": f.get("tli", float("nan")),
                "rmsea": f.get("rmsea", float("nan")),
                "srmr": f.get("srmr", float("nan")),
                "delta_cfi": delta.get("delta_cfi", float("nan")),
                "delta_rmsea": delta.get("delta_rmsea", float("nan")),
                "passed": passed,
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["highest_invariance"] = highest
    return df


def cheatsheet() -> str:
    return "mi_summary({}) -> Summary table of invariance levels with pass/fail."

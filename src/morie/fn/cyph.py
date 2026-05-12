# morie.fn -- function file (hadesllm/morie)
"""Difficulties strengthen the mind, as labor does the body. -- Seneca"""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def detect_leakage(
    data: pd.DataFrame,
    *,
    target: str = "outcome",
    threshold: float = 0.95,
) -> DescriptiveResult:
    """Detect potential data leakage by identifying features with
    suspiciously high correlation to the target variable.

    Checks for:
    1. Perfect or near-perfect correlations (Pearson |r| > threshold).
    2. Features that are deterministic transforms of the target.
    3. Future-information proxies (columns whose name contains the target name).

    Parameters
    ----------
    data : DataFrame
        Input dataset.
    target : str
        Target column name.
    threshold : float
        Absolute correlation above which a feature is flagged (default 0.95).

    Returns
    -------
    DescriptiveResult
        ``value`` is the number of leaked features; ``extra`` has details.
    """
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found")

    numeric = data.select_dtypes(include="number")
    if target not in numeric.columns:
        raise ValueError(f"Target '{target}' must be numeric")

    features = [c for c in numeric.columns if c != target]
    if not features:
        return DescriptiveResult(
            name="LeakageDetection",
            value=0,
            extra={"flagged": [], "correlations": {}},
        )

    y = numeric[target]
    corrs = {}
    flagged_corr = []
    for f in features:
        r = float(y.corr(numeric[f]))
        corrs[f] = r
        if abs(r) > threshold:
            flagged_corr.append({"feature": f, "correlation": r, "reason": "high_correlation"})

    flagged_name = []
    target_lower = target.lower()
    for c in data.columns:
        if c == target:
            continue
        if target_lower in c.lower() and c in features:
            flagged_name.append({"feature": c, "reason": "name_contains_target"})

    flagged_determ = []
    for f in features:
        col = numeric[f].dropna()
        tgt = y.loc[col.index]
        if len(col) > 10:
            unique_ratios = col.nunique()
            if unique_ratios <= 2 and tgt.nunique() <= 2:
                acc = float(((col > col.median()) == (tgt > tgt.median())).mean())
                if acc > threshold or acc < (1 - threshold):
                    flagged_determ.append({"feature": f, "accuracy": acc, "reason": "deterministic"})

    all_flagged = flagged_corr + flagged_name + flagged_determ
    seen = set()
    unique_flagged = []
    for item in all_flagged:
        if item["feature"] not in seen:
            seen.add(item["feature"])
            unique_flagged.append(item)

    return DescriptiveResult(
        name="LeakageDetection",
        value=len(unique_flagged),
        extra={
            "flagged": unique_flagged,
            "correlations": corrs,
            "threshold": threshold,
            "n_features": len(features),
        },
    )


cyph = detect_leakage


def cheatsheet() -> str:
    return "Difficulties strengthen the mind, as labor does the body. -- Seneca"

"""It does not matter how slowly you go as long as you do not stop. — Confucius"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def record_linkage(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    on: list[str] | None = None,
    threshold: float = 0.8,
) -> DescriptiveResult:
    """Probabilistic record linkage between two DataFrames.

    Computes Jaro-Winkler-like similarity on shared string columns and
    exact match on numeric columns. Returns pairs that exceed *threshold*.

    Parameters
    ----------
    df1, df2 : DataFrame
        Two datasets to link.
    on : list of str or None
        Columns to compare (must exist in both). Defaults to shared columns.
    threshold : float
        Minimum composite similarity score (0--1) to declare a match.

    Returns
    -------
    DescriptiveResult
        ``value`` is the number of matched pairs; ``extra`` has the
        match pairs with scores.

    References
    ----------
    Fellegi, I. P. & Sunter, A. B. (1969). A theory for record linkage.
    JASA, 64(328), 1183-1210.
    """
    if on is None:
        on = [c for c in df1.columns if c in df2.columns]
    if not on:
        raise ValueError("No shared columns found")
    for c in on:
        if c not in df1.columns or c not in df2.columns:
            raise ValueError(f"Column '{c}' missing from one of the DataFrames")

    def _jaro(s1, s2):
        s1, s2 = str(s1), str(s2)
        if s1 == s2:
            return 1.0
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        window = max(len1, len2) // 2 - 1
        if window < 0:
            window = 0
        match1 = [False] * len1
        match2 = [False] * len2
        matches = 0
        transpositions = 0
        for i in range(len1):
            lo = max(0, i - window)
            hi = min(len2, i + window + 1)
            for j in range(lo, hi):
                if match2[j] or s1[i] != s2[j]:
                    continue
                match1[i] = match2[j] = True
                matches += 1
                break
        if matches == 0:
            return 0.0
        k = 0
        for i in range(len1):
            if not match1[i]:
                continue
            while not match2[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

    matches = []
    for i in range(len(df1)):
        for j in range(len(df2)):
            scores = []
            for c in on:
                v1 = df1[c].iloc[i]
                v2 = df2[c].iloc[j]
                if pd.api.types.is_numeric_dtype(df1[c]):
                    scores.append(1.0 if v1 == v2 else 0.0)
                else:
                    scores.append(_jaro(v1, v2))
            composite = float(np.mean(scores))
            if composite >= threshold:
                matches.append(
                    {
                        "idx_1": int(df1.index[i]),
                        "idx_2": int(df2.index[j]),
                        "score": round(composite, 4),
                    }
                )

    return DescriptiveResult(
        name="Record Linkage",
        value=len(matches),
        extra={
            "matches": matches[:200],
            "threshold": threshold,
            "columns": on,
            "n1": len(df1),
            "n2": len(df2),
        },
    )


thmss = record_linkage


def cheatsheet() -> str:
    return "It does not matter how slowly you go as long as you do not stop. — Confucius"

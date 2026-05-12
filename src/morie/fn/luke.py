# morie.fn -- function file (hadesllm/morie)
"""What is now proved was once only imagined. -- William Blake"""

from __future__ import annotations

import pandas as pd
from scipy import stats as _st

from ._containers import DescriptiveResult


def summarize(data: pd.DataFrame) -> DescriptiveResult:
    """Per-column summary: n, mean, std, min, q25, median, q75, max, missing, skew."""
    rows = []
    for col in data.columns:
        s = data[col]
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            rows.append(
                {
                    "column": col,
                    "n": len(clean),
                    "missing": int(s.isna().sum()),
                    "mean": float(clean.mean()),
                    "std": float(clean.std()),
                    "min": float(clean.min()),
                    "q25": float(clean.quantile(0.25)),
                    "median": float(clean.median()),
                    "q75": float(clean.quantile(0.75)),
                    "max": float(clean.max()),
                    "skew": float(_st.skew(clean)) if len(clean) > 2 else float("nan"),
                }
            )
        else:
            rows.append(
                {
                    "column": col,
                    "n": int(s.notna().sum()),
                    "missing": int(s.isna().sum()),
                    "n_unique": int(s.nunique()),
                    "mode": str(s.mode().iloc[0]) if len(s.mode()) > 0 else None,
                }
            )
    return DescriptiveResult(name="Summary", value=pd.DataFrame(rows))


luke = summarize


def cheatsheet() -> str:
    return "summarize({}) -> Per-column summary stats. 'I've got a bad feeling about this"

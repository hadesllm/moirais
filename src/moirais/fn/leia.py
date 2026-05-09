# moirais.fn — function file (hadesllm/moirais)
"""Dataset profiling. 'Knowing others is intelligence; knowing yourself is true wisdom. — Lao Tzu'

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def profile(data: pd.DataFrame) -> DescriptiveResult:
    """Profile a dataset: shape, types, missingness, correlations."""
    n, p = data.shape
    types = {col: str(data[col].dtype) for col in data.columns}
    missing = {col: int(data[col].isna().sum()) for col in data.columns}
    pct_missing = {col: round(v / n * 100, 1) for col, v in missing.items()}
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    high_corr = []
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()
        for i, c1 in enumerate(numeric_cols):
            for j, c2 in enumerate(numeric_cols):
                if i < j and abs(corr.iloc[i, j]) > 0.7:
                    high_corr.append({"col1": c1, "col2": c2, "r": round(float(corr.iloc[i, j]), 3)})
    return DescriptiveResult(
        name="Profile",
        value={"n_rows": n, "n_cols": p},
        extra={
            "types": types,
            "missing": missing,
            "pct_missing": pct_missing,
            "n_numeric": len(numeric_cols),
            "high_correlations": high_corr,
        },
    )


 = profile


def cheatsheet() -> str:
    return "profile({}) -> Dataset profiling. 'Knowing others is intelligence; knowing yourself is true wisdom. — Lao Tzu'

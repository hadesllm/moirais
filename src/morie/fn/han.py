# morie.fn — function file (hadesllm/morie)
"""Missing data handler. 'The unexamined life is not worth living. -- Socrates'"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def handle_missing(data: pd.DataFrame, *, method: str = "summary") -> DescriptiveResult:
    """Missing data: 'summary' reports patterns, 'mean'/'median'/'drop' imputes."""
    n, p = data.shape
    missing = {col: int(data[col].isna().sum()) for col in data.columns}
    total_missing = sum(missing.values())
    if method == "summary":
        return DescriptiveResult(
            name="Missing data summary",
            value={"total_missing": total_missing, "pct": round(total_missing / (n * p) * 100, 1)},
            extra={
                "per_column": missing,
                "complete_rows": int(data.dropna().shape[0]),
                "n_rows": n,
                "n_cols": p,
            },
        )
    elif method == "drop":
        clean = data.dropna()
        return DescriptiveResult(
            name="Drop missing",
            value=clean,
            extra={"rows_before": n, "rows_after": len(clean), "dropped": n - len(clean)},
        )
    elif method in ("mean", "median"):
        result = data.copy()
        for col in result.select_dtypes(include=[np.number]).columns:
            fill = result[col].mean() if method == "mean" else result[col].median()
            result[col] = result[col].fillna(fill)
        return DescriptiveResult(
            name=f"Impute ({method})",
            value=result,
            extra={"imputed_count": total_missing, "method": method},
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'summary', 'mean', 'median', or 'drop'.")


han = handle_missing


def cheatsheet() -> str:
    return "handle_missing({}) -> Missing data handler. 'Never tell me the odds.' -- Han Solo"

# morie.fn — function file (hadesllm/morie)
"""Summary of flagged DIF items across methods."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult, DIFResult


def dif_flag_summary(
    *results: DIFResult,
    method_names: list[str] | None = None,
) -> DescriptiveResult:
    """Summarise flagged items across multiple DIF methods.

    Parameters
    ----------
    *results : DIFResult
        Multiple DIF analysis results.
    method_names : list[str], optional
        Names for each method. Default uses result.method.

    Returns
    -------
    DescriptiveResult
        value=DataFrame showing which items flagged by which methods.
    """
    if not results:
        return DescriptiveResult(name="DIF flag summary", value=pd.DataFrame())

    if method_names is None:
        method_names = [r.method for r in results]

    all_items = set()
    for r in results:
        all_items.update(r.flagged)
        if hasattr(r.items, "iterrows"):
            for _, row in r.items.iterrows():
                col = "item" if "item" in row.index else "bundle"
                if col in row.index:
                    all_items.add(row[col])

    all_items = sorted(all_items)
    rows = []
    for item in all_items:
        row = {"item": item}
        n_flagged = 0
        for name, res in zip(method_names, results):
            is_flagged = item in res.flagged
            row[name] = is_flagged
            if is_flagged:
                n_flagged += 1
        row["n_methods_flagged"] = n_flagged
        rows.append(row)

    df = pd.DataFrame(rows)
    return DescriptiveResult(
        name="DIF flag summary",
        value=df,
        extra={
            "n_methods": len(results),
            "n_items_flagged": len([r for r in rows if r["n_methods_flagged"] > 0]),
        },
    )


flag_summary = dif_flag_summary


def cheatsheet() -> str:
    return "dif_flag_summary({}) -> Summary of flagged DIF items across methods."

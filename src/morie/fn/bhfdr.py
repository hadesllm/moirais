# morie.fn -- function file (hadesllm/morie)
"""Benjamini-Hochberg FDR with R-style verbose result."""

from typing import Sequence


def bhfdr(p_values: Sequence[float], alpha: float = 0.05):
    """BH FDR rejection vector."""
    from ._richresult import RichResult
    m = len(p_values)
    if m == 0:
        return RichResult(title="Benjamini-Hochberg FDR", summary_lines=[], payload={"rejects": []})
    order = sorted(range(m), key=lambda i: p_values[i])
    cutoff_k = -1
    for k, idx in enumerate(order, start=1):
        if p_values[idx] <= k * alpha / m:
            cutoff_k = k
    out = [False] * m
    if cutoff_k > 0:
        for k_idx in range(cutoff_k):
            out[order[k_idx]] = True
    rows = []
    for k, idx in enumerate(order, start=1):
        thresh = k * alpha / m
        rows.append([k, idx, f"{p_values[idx]:.4g}", f"{thresh:.4g}",
                     "REJECT" if out[idx] else "fail to reject"])
    n_reject = sum(out)
    return RichResult(
        title="Benjamini-Hochberg false discovery rate (FDR)",
        summary_lines=[
            ("Number of tests m", m),
            ("FDR target", alpha),
            ("Rejected", f"{n_reject} of {m}"),
        ],
        tables=[{
            "title": "Per-test results (sorted by p-value):",
            "headers": ["Rank", "Test idx", "p-value", "Threshold (k*a/m)", "Decision"],
            "rows": rows,
        }],
        interpretation=(f"BH controls expected FDR at {alpha}, NOT FWER -- more "
                        "powerful than Bonferroni but allows some false positives "
                        "in rejection set."),
        payload={"rejects": out, "n_reject": n_reject, "alpha": alpha},
    )

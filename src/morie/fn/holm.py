# morie.fn — function file (hadesllm/morie)
"""Holm-Bonferroni step-down with R-style verbose result."""

from typing import Sequence


def holm(p_values: Sequence[float], alpha: float = 0.05):
    """Holm-Bonferroni step-down rejection vector."""
    from ._richresult import RichResult
    m = len(p_values)
    if m == 0:
        return RichResult(title="Holm step-down", summary_lines=[], payload={"rejects": []})
    order = sorted(range(m), key=lambda i: p_values[i])
    rejects_sorted = []
    can_reject = True
    for k, idx in enumerate(order):
        thresh = alpha / (m - k)
        if can_reject and p_values[idx] <= thresh:
            rejects_sorted.append(True)
        else:
            rejects_sorted.append(False)
            can_reject = False
    out = [False] * m
    for k, idx in enumerate(order):
        out[idx] = rejects_sorted[k]
    rows = []
    for k, idx in enumerate(order):
        thresh = alpha / (m - k)
        rows.append([k+1, idx, f"{p_values[idx]:.4g}", f"{thresh:.4g}",
                     "REJECT" if out[idx] else "fail to reject"])
    n_reject = sum(out)
    return RichResult(
        title="Holm-Bonferroni step-down rejection",
        summary_lines=[
            ("Number of tests m", m),
            ("FWER alpha", alpha),
            ("Rejected", f"{n_reject} of {m}"),
        ],
        tables=[{
            "title": "Per-test results (sorted by p-value):",
            "headers": ["Rank", "Test idx", "p-value", "Threshold", "Decision"],
            "rows": rows,
        }],
        interpretation=(f"Holm controls FWER at {alpha}. Steps down: smallest "
                        "p compared to alpha/m; once a test fails, all larger "
                        "p-values automatically fail."),
        payload={"rejects": out, "n_reject": n_reject, "alpha": alpha},
    )

# morie.fn -- function file (hadesllm/morie)
"""Compare W-NOMINATE vs DW-NOMINATE scores."""

from __future__ import annotations

from ._containers import DescriptiveResult


def nominate_score_types(X_w, X_dw) -> DescriptiveResult:
    """Compare W-NOMINATE and DW-NOMINATE ideal point estimates.

    .. epigraph:: "Stay out of my territory." -- Walter White, Breaking Bad
    """
    import numpy as np

    w = np.asarray(X_w, dtype=float)
    dw = np.asarray(X_dw, dtype=float)
    corr = float(np.corrcoef(w.ravel(), dw.ravel())[0, 1])
    diff = w.ravel() - dw.ravel()
    return DescriptiveResult(
        name="nominate_score_types",
        value=corr,
        extra={
            "correlation": corr,
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "n": len(w.ravel()),
        },
    )


nosco = nominate_score_types


def cheatsheet() -> str:
    return "nominate_score_types({}) -> Compare W-NOMINATE vs DW-NOMINATE scores."

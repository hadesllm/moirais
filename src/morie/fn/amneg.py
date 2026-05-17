# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Detect A-M negative-weight (reversed-ordering) respondents."""

from __future__ import annotations

from ._containers import DescriptiveResult


def am_negative_weights(beta) -> DescriptiveResult:
    """Identify respondents with negative weights (reversed ordering).

    :param beta: Per-respondent weight vector from A-M.
    :return: DescriptiveResult with indices of negative-weight respondents.

    .. epigraph:: There is no royal road to geometry. -- Euclid
    """
    import numpy as np

    beta = np.asarray(beta, dtype=float).ravel()
    neg_idx = np.where(beta < 0)[0]
    return DescriptiveResult(
        name="am_negative_weights",
        value=len(neg_idx),
        extra={
            "negative_indices": neg_idx.tolist(),
            "n_negative": len(neg_idx),
            "n_total": len(beta),
            "pct_negative": float(len(neg_idx) / len(beta) * 100) if len(beta) > 0 else 0.0,
        },
    )


amneg = am_negative_weights


def cheatsheet() -> str:
    return "am_negative_weights({}) -> Detect A-M negative-weight (reversed-ordering) respondents."

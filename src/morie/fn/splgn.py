"""We suffer more often in imagination than in reality. — Seneca"""

from __future__ import annotations

from ._containers import DescriptiveResult


def nasa_tlx(
    *,
    mental: float = 50.0,
    physical: float = 50.0,
    temporal: float = 50.0,
    performance: float = 50.0,
    effort: float = 50.0,
    frustration: float = 50.0,
    weights: dict[str, float] | None = None,
) -> DescriptiveResult:
    """Compute NASA Task Load Index (NASA-TLX) score.

    The raw TLX is the unweighted mean of six subscales (each 0--100).
    Weighted TLX uses pairwise comparison weights.

    Parameters
    ----------
    mental, physical, temporal, performance, effort, frustration : float
        Subscale ratings (0--100).
    weights : dict or None
        Optional weights for each subscale (keys must match subscale names).
        If None, computes raw (unweighted) TLX.

    Returns
    -------
    DescriptiveResult
        ``value`` is the overall TLX score; ``extra`` has subscale values.

    References
    ----------
    Hart, S. G. & Staveland, L. E. (1988). Development of NASA-TLX.
    Advances in Psychology, 52, 139-183.
    """
    scales = {
        "mental": mental,
        "physical": physical,
        "temporal": temporal,
        "performance": performance,
        "effort": effort,
        "frustration": frustration,
    }
    for name, val in scales.items():
        if not (0 <= val <= 100):
            raise ValueError(f"{name} must be in [0, 100], got {val}")

    if weights is None:
        score = sum(scales.values()) / 6.0
        method = "raw"
    else:
        for k in weights:
            if k not in scales:
                raise ValueError(f"Unknown weight key '{k}'")
        w_total = sum(weights.values())
        if w_total <= 0:
            raise ValueError("Weights must sum to a positive value")
        score = sum(scales[k] * weights.get(k, 0) for k in scales) / w_total
        method = "weighted"

    return DescriptiveResult(
        name="NASA-TLX",
        value=round(score, 2),
        extra={**scales, "method": method},
    )


splgn = nasa_tlx


def cheatsheet() -> str:
    return "nasa_tlx({}) -> NASA-TLX cognitive load metric. 'You have the look of a man "

"""Template-match event detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def template_match_detect(x: np.ndarray, template: np.ndarray, threshold: float = 0.7) -> DescriptiveResult:
    """Knowing yourself is the beginning of all wisdom. — Aristotle"""
    from morie._detection import template_match as _backend

    indices, corr = _backend(x, template, threshold=threshold)
    return DescriptiveResult(
        name="template_match",
        value=int(len(indices)),
        extra={"indices": indices, "correlations": corr},
    )


alias = template_match_detect


def cheatsheet() -> str:
    return "template_match_detect({}) -> Template-match event detection."

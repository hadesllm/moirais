# morie.fn — function file (hadesllm/morie)
"""Infer Stevens measurement level (NOIR) for a column."""

from morie.dataset import infer_measurement_level as _fn

infml = _fn
infer_measurement_level = _fn


def cheatsheet() -> str:
    return "infml() -> Infer Stevens measurement level (NOIR) for a column."

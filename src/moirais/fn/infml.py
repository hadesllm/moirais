# moirais.fn — function file (hadesllm/moirais)
"""Infer Stevens measurement level (NOIR) for a column."""

from moirais.dataset import infer_measurement_level as _fn

infml = _fn
infer_measurement_level = _fn


def cheatsheet() -> str:
    return "infml() -> Infer Stevens measurement level (NOIR) for a column."

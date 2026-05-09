# moirais.fn — function file (hadesllm/moirais)
"""Inspect a single output file (schema, row counts, stats)."""

from moirais.inspector import inspect_output as _fn

inspct = _fn
inspect_output = _fn


def cheatsheet() -> str:
    return "inspct() -> Inspect a single output file (schema, row counts, stats)."

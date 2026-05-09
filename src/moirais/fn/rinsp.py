# moirais.fn — function file (hadesllm/moirais)
"""Render an inspection result to the terminal."""

from moirais.inspector import render_inspection as _fn

rinsp = _fn
render_inspection = _fn


def cheatsheet() -> str:
    return "rinsp() -> Render an inspection result to the terminal."

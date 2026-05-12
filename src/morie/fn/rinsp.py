# morie.fn -- function file (hadesllm/morie)
"""Render an inspection result to the terminal."""

from morie.inspector import render_inspection as _fn

rinsp = _fn
render_inspection = _fn


def cheatsheet() -> str:
    return "rinsp() -> Render an inspection result to the terminal."

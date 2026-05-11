# morie.fn — function file (hadesllm/morie)
"""Render a verification report to the terminal."""

from morie.inspector import render_verification as _fn

rvrfy = _fn
render_verification = _fn


def cheatsheet() -> str:
    return "rvrfy() -> Render a verification report to the terminal."

# moirais.fn — function file (hadesllm/moirais)
"""Render a verification report to the terminal."""

from moirais.inspector import render_verification as _fn

rvrfy = _fn
render_verification = _fn


def cheatsheet() -> str:
    return "rvrfy() -> Render a verification report to the terminal."

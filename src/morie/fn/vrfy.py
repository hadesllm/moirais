"""Verify statistical output: p-values, CIs, SEs, ORs."""

from morie.inspector import verify_statistical_output as _fn

vrfy = _fn
verify_statistical_output = _fn


def cheatsheet() -> str:
    return "vrfy() -> Verify statistical output: p-values, CIs, SEs, ORs."

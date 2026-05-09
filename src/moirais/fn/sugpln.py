"""Suggest an analysis plan from a dataset profile."""

from moirais.dataset import suggest_analysis_plan as _fn

sugpln = _fn
suggest_analysis_plan = _fn


def cheatsheet() -> str:
    return "sugpln() -> Suggest an analysis plan from a dataset profile."

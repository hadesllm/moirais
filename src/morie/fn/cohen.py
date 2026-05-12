# morie.fn -- function file (hadesllm/morie)
"""Cohen's d effect size."""

from morie.fn.d import cohens_d

cohen = cohens_d


def cheatsheet() -> str:
    return "cohen() -> Cohen's d effect size."

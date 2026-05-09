# moirais.fn — function file (hadesllm/moirais)
"""Cohen's d effect size."""

from moirais.fn.d import cohens_d

cohen = cohens_d


def cheatsheet() -> str:
    return "cohen() -> Cohen's d effect size."

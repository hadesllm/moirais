"""Wilcoxon signed-rank test."""

from moirais.fn.wilcox import wilcoxon_signed_rank_test

wlcx = wilcoxon_signed_rank_test
wilcoxon = wilcoxon_signed_rank_test


def cheatsheet() -> str:
    return "wlcx() -> Wilcoxon signed-rank test."

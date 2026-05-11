"""Mann-Whitney U test."""

from morie.fn.mw import mann_whitney_test

utest = mann_whitney_test
mann_whitney = mann_whitney_test


def cheatsheet() -> str:
    return "utest() -> Mann-Whitney U test."

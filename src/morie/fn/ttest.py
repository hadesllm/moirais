"""Student's t-test."""

from morie.fn.t2smp import two_sample_t_test

ttest = two_sample_t_test
t_test = two_sample_t_test


def cheatsheet() -> str:
    return "ttest() -> Student's t-test."

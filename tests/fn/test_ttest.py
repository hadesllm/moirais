"""Tests for moirais.fn.ttest -- t-test alias."""

from moirais.fn.ttest import ttest, t_test
from moirais.fn.t2smp import two_sample_t_test


class TestTtest:
    def test_alias_imports(self):
        assert callable(ttest)

    def test_ttest_is_two_sample_t_test(self):
        assert ttest is two_sample_t_test

    def test_t_test_alias(self):
        assert t_test is two_sample_t_test

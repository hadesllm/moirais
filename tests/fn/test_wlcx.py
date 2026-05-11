"""Tests for morie.fn.wlcx -- Wilcoxon alias."""

from morie.fn.wlcx import wlcx, wilcoxon
from morie.fn.wilcox import wilcoxon_signed_rank_test


class TestWlcx:
    def test_alias_imports(self):
        assert callable(wlcx)

    def test_wlcx_is_wilcoxon_signed_rank_test(self):
        assert wlcx is wilcoxon_signed_rank_test

    def test_wilcoxon_alias(self):
        assert wilcoxon is wilcoxon_signed_rank_test

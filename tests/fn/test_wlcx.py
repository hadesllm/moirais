"""Tests for moirais.fn.wlcx -- Wilcoxon alias."""

from moirais.fn.wlcx import wlcx, wilcoxon
from moirais.fn.wilcox import wilcoxon_signed_rank_test


class TestWlcx:
    def test_alias_imports(self):
        assert callable(wlcx)

    def test_wlcx_is_wilcoxon_signed_rank_test(self):
        assert wlcx is wilcoxon_signed_rank_test

    def test_wilcoxon_alias(self):
        assert wilcoxon is wilcoxon_signed_rank_test

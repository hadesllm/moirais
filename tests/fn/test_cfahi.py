"""Tests for cfahi -- Higher-order CFA."""
import numpy as np
from morie.fn.cfahi import cfa_hierarchical
from morie.fn._containers import DescriptiveResult


class TestCfaHierarchical:
    def test_basic(self, mapq_df):
        result = cfa_hierarchical(mapq_df)
        assert isinstance(result, DescriptiveResult)
        assert "first_order_loadings" in result.value

    def test_second_order(self, mapq_df):
        result = cfa_hierarchical(mapq_df)
        so = result.value["second_order_loadings"]
        assert len(so) == 4

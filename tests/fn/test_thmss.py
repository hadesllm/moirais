"""Tests for morie.fn.thmss -- record linkage."""

import pandas as pd
from morie.fn.thmss import record_linkage, thmss
from morie.fn._containers import DescriptiveResult


class TestThmss:
    def test_alias(self):
        assert thmss is record_linkage

    def test_exact_match(self):
        df1 = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
        df2 = pd.DataFrame({"name": ["Alice", "Charlie"], "age": [30, 40]})
        result = record_linkage(df1, df2, threshold=0.8)
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 1

    def test_no_match(self):
        df1 = pd.DataFrame({"name": ["AAA"], "age": [10]})
        df2 = pd.DataFrame({"name": ["ZZZ"], "age": [99]})
        result = record_linkage(df1, df2, threshold=0.9)
        assert result.value == 0

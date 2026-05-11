"""Tests for morie.fn.blpil -- naive baseline estimator."""

import pandas as pd
from morie.fn.blpil import naive_baseline, blpil
from morie.fn._containers import DescriptiveResult


class TestBlpil:
    def test_alias(self):
        assert blpil is naive_baseline

    def test_mean_baseline(self):
        df = pd.DataFrame({"outcome": [1, 2, 3, 4, 5]})
        result = naive_baseline(df, target="outcome", method="mean")
        assert isinstance(result, DescriptiveResult)
        assert result.value == 3.0

    def test_mode_baseline(self):
        df = pd.DataFrame({"outcome": [0, 0, 0, 1, 1]})
        result = naive_baseline(df, target="outcome", method="mode")
        assert result.value == 0
        assert result.extra["accuracy"] == 0.6

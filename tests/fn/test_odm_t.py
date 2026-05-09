"""Tests for moirais.fn.odm_t — OTIS demo trend."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.odm_t import otis_demo_trend
from moirais.fn._containers import DescriptiveResult


class TestOtisDemoTrend:

    def test_returns_descriptive(self):
        df = pd.DataFrame({"group": ["A", "B", "A", "B", "A", "B"],
                           "period": [1, 1, 1, 2, 2, 2]})
        result = otis_demo_trend(df)
        assert isinstance(result, DescriptiveResult)

    def test_proportions_sum_one(self):
        df = pd.DataFrame({"group": ["X", "Y", "X", "Y"], "period": [1, 1, 2, 2]})
        result = otis_demo_trend(df)
        for _, row in result.value.iterrows():
            assert row.sum() == pytest.approx(1.0)

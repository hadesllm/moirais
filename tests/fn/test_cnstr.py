"""Tests for moirais.fn.cnstr -- IQR anomaly removal."""

import numpy as np
import pandas as pd
from moirais.fn.cnstr import iqr_exorcise, cnstr
from moirais.fn._containers import DescriptiveResult


class TestCnstr:
    def test_alias(self):
        assert cnstr is iqr_exorcise

    def test_removes_outliers(self):
        x = list(range(20)) + [1000]
        df = pd.DataFrame({"x": x})
        result = iqr_exorcise(df, col="x")
        assert isinstance(result, DescriptiveResult)
        assert result.value >= 1
        assert 1000 in result.extra["removed_values"]

    def test_no_removal(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = iqr_exorcise(df, factor=3.0)
        assert result.value == 0

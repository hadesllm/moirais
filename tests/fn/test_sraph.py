"""Tests for moirais.fn.sraph -- input validation."""

import numpy as np
import pandas as pd
from moirais.fn.sraph import validate_inputs, sraph
from moirais.fn._containers import DescriptiveResult


class TestSraph:
    def test_alias(self):
        assert sraph is validate_inputs

    def test_clean_data(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = validate_inputs(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 1.0

    def test_missing_data(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
        result = validate_inputs(df)
        assert result.value < 1.0
        assert result.extra["violations"]["a"]["missing"] == 1

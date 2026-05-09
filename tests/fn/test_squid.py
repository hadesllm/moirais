"""Tests for moirais.fn.squid -- threat scoring."""

import numpy as np
import pandas as pd
from moirais.fn.squid import threat_score, squid
from moirais.fn._containers import DescriptiveResult


class TestSquid:
    def test_alias(self):
        assert squid is threat_score

    def test_basic(self):
        df = pd.DataFrame({"risk1": [1, 2, 3, 4, 5], "risk2": [5, 4, 3, 2, 1]})
        result = threat_score(df)
        assert isinstance(result, DescriptiveResult)
        assert 0 <= result.value <= 1

    def test_single_feature(self):
        df = pd.DataFrame({"x": [10, 20, 30]})
        result = threat_score(df, features=["x"])
        assert result.extra["n_features"] == 1

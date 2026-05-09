"""Tests for moirais.fn.tpsmj — major crime."""

import pytest
import pandas as pd
from moirais.fn.tpsmj import tps_major_crime
from moirais.fn._containers import DescriptiveResult


class TestMajorCrime:
    def test_basic(self):
        df = pd.DataFrame({"mci_category": ["Assault"] * 10 + ["Robbery"] * 5, "count": [1] * 15})
        r = tps_major_crime(df)
        assert isinstance(r, DescriptiveResult)
        assert r.value == pytest.approx(15.0)

    def test_proportions(self):
        df = pd.DataFrame({"mci_category": ["A"] * 3 + ["B"] * 7})
        r = tps_major_crime(df)
        assert r.extra["proportions"]["B"] == pytest.approx(0.7)
